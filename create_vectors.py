import pyodbc
from tensorflow.keras.preprocessing import image
import requests
import numpy as np
from scipy import spatial
from sklearn.preprocessing import minmax_scale
import json
import time
import logging

# Gets or creates a logger
logger = logging.getLogger(__name__)
# set log level
logger.setLevel(logging.DEBUG)
# define file handler and set formatter
file_handler = logging.FileHandler('log.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add file handler to logger
logger.addHandler(file_handler)

conn = None
GAP = 0  # seconds to sleep between the loop steps

# Model REST API - tf serving - predict service URL
tf_serving_url = 'http://localhost:8501/v1/models/similarityModel:predict'
headers = {"content-type": "application/json"}

# mount path to access the file Server
fs = "/mnt/muesfs/mues-images/image/ak/"

# MILVUS REST API URL
milvus_url = 'http://localhost:19121/collections/artifact/vectors'


def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    img = image.img_to_array(img)
    # img = np.array(img)
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    return img


def connect_to_db():
    global conn
    try:
        conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};"
                              "SERVER=172.17.20.41;PORT=1433;UID=muesd;PWD=Mues*dev.1;DATABASE=mues_dev")
        logger.info('DB connected successfully')
    except Exception as e:
        logger.critical('DB Connection Error: ' + e)


def create_top_n_vectors():
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 100 F.ESER_ID, F.FOTOGRAF_PATH FROM ESER_FOTOGRAF F "
                   "LEFT JOIN ESER E ON F.ESER_ID = E.ID "
                   "WHERE ANA_FOTOGRAF=1 AND DOLASIM_KOPYASI_PATH is NULL AND "
                   "E.AKTIF=1 AND E.SILINMIS=0 order by F.ESER_ID")

    records = cursor.fetchall()

    ids = []
    vectors = []
    db_result_dic = dict()

    for row in records:
        try:
            logger.info("id: " + str(row[0]) + " : " + str(row[1]))
            # print(("id: " + str(row[0]) + " : " + str(row[1])))

            img = image.load_img(fs + row[1])
            img_data = prepare_image(img)

            # prepare for tf serving service
            # give the photo and get the vector from the model
            data = json.dumps({"signature_name": "serving_default", "instances": img_data.tolist()})
            response = requests.post(tf_serving_url, data=data, headers=headers)
            dict_resp = json.loads(response.text)
            feature_np = np.array(dict_resp["predictions"])
            # min-max scale the data between 0 and 1
            scaled_vec = minmax_scale(feature_np.flatten())
            result_vec = np.round(scaled_vec, 2)
            # print(result_vec)

            # for milvus request
            ids.append(str(row[0]))
            vectors.append(result_vec.tolist())

            db_result_dic[str(row[0])] = "1"

        except (FileNotFoundError, IOError):
            logger.error("File not found: " + fs + row[1])
            db_result_dic[str(row[0])] = "-1" # marking for FileNotFound
        except ValueError as e:
            logger.error("Decoding JSON has failed: " + e)
        except (requests.HTTPError, requests.RequestException) as e:
            logger.error("HTTP/Request error occurred: " + e)

    try:
        # save the n vector to the Milvus DB
        data_milvus = json.dumps({"ids": ids, "vectors": vectors})
        resp_milvus = requests.post(milvus_url, data=data_milvus, headers=headers)
        # logger.info(resp_milvus)
    except Exception as e:
        logger.error("MILVUS post request error: " + e)

    try:
        # commit for top N selected records
        dict_ok = {key: value for (key, value) in db_result_dic.items() if value == '1'}
        dict_err = {key: value for (key, value) in db_result_dic.items() if value == '-1'}

        ok_list = tuple(dict_ok.keys())
        conn.execute(
            "UPDATE ESER_FOTOGRAF set DOLASIM_KOPYASI_PATH='1' where ANA_FOTOGRAF=1 AND ESER_ID in {}".format(ok_list));

        err_list = tuple(dict_err.keys())
        conn.execute(
            "UPDATE ESER_FOTOGRAF set DOLASIM_KOPYASI_PATH='-1' where ANA_FOTOGRAF=1 AND ESER_ID in {}".format(err_list));

        conn.commit()

    except Exception as e:
        logger.error(e + " --> Trying to reconnect to the DB...")
        conn.close()
        connect_to_db()

    return len(records)


def create_all():
    while True:
        records_len = create_top_n_vectors()
        logger.info(str(records_len) + " vectors created successfully")
        if records_len == 0:
            logger.info("No record found to get the vector")
        time.sleep(GAP)


if __name__ == "__main__":
    connect_to_db()
    create_all()
