import pymssql
from PIL import Image
import requests
import numpy as np
from scipy import spatial
from sklearn.preprocessing import minmax_scale
import json
import time
import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)


# Gets or creates a logger
logger = logging.getLogger(__name__)
# set log level
logger.setLevel(logging.DEBUG)
# define file handler and set formatter
file_handler = logging.FileHandler('log_create_vector.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add file handler to logger
logger.addHandler(file_handler)

conn = None
GAP = 2  # seconds to sleep between the loop steps

# Model REST API - tf serving - predict service URL
# tf_serving_url = 'http://localhost:8501/v1/models/similarityModel:predict'
tf_serving_url = 'http://localhost:8501/v1/models/resnet50:predict'
headers = {"content-type": "application/json"}

# mount path to access the file Server
# fs = "/mnt/muesfs/mues-images/image/ak/" # prod
fs = "/mnt/muesfs/mues/mues-images/dev/image/ak/" # dev

# QDRANT REST API URL
qdrant_url = 'http://localhost:6333/collections/artifact/points'


def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    # img = image.img_to_array(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    return img


def connect_to_db():
    global conn
    try:
        conn = pymssql.connect(server='10.1.37.177', port='1033', user='muest', password='Mues*test.1', database='mues_test')
        logger.info('DB connected successfully')
    except Exception as e:
        logger.critical(e)


def create_top_n_vectors():
    cursor = conn.cursor()
    cursor.execute("select DISTINCT TOP 100 F.ESER_ID, F.FOTOGRAF_PATH from ESER_FOTOGRAF F "
                   "LEFT JOIN ESER E ON F.ESER_ID=E.ID "
                   "WHERE permanentId is not NULL AND E.AKTIF=1 AND E.SILINMIS=0 AND F.ANA_FOTOGRAF=1 AND F.FEATURE_VECTOR_STATE is NULL ORDER BY F.ESER_ID")

    records = cursor.fetchall()

    ids = []
    vectors = []
    artifact_types = []
    ok_list = []
    err_list = []

    for row in records:
        try:
            logger.info("id:" + str(row[0]) + " : " + str(row[1]))
            print(("id: " + str(row[0]) + " : " + str(row[1])))

            img = Image.open(fs + row[1])
            # img = image.load_img(fs + row[1])
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
            ids.append(row[0])

            artifact_types.append({"artifactType":1})

            vectors.append(result_vec.tolist())

            ok_list.append(str(row[0]))

        except (FileNotFoundError, IOError):
            logger.error("File not found: " + fs + row[1])
            err_list.append(str(row[0]))  # marking for FileNotFound
        except ValueError as e:
            logger.error("Decoding JSON has failed")
            logger.error(e)
        except (requests.HTTPError, requests.RequestException) as e:
            logger.error("HTTP/Request error occurred")
            logger.error(e)

    try:
        # save the n vector to the Qdrant
        if(len(vectors) > 0):
            data_json = json.dumps({"batch": {"ids":ids, "payloads":artifact_types, "vectors":vectors} })
            # logger.info(data_json)

            response = requests.put(qdrant_url, data=data_json, headers=headers)
    except Exception as e:
        logger.error("QDRANT put request error")
        logger.error(e)

    try:
        # commit for top N selected records
        if(len(ok_list)>0):
            cursor.execute("UPDATE ESER_FOTOGRAF set FEATURE_VECTOR_STATE='1' where ANA_FOTOGRAF=1 AND ESER_ID in {}".format(str(tuple(ok_list)).replace(',)', ')')))
        if(len(err_list)>0):
            cursor.execute("UPDATE ESER_FOTOGRAF set FEATURE_VECTOR_STATE='-1' where ANA_FOTOGRAF=1 AND ESER_ID in {}".format(str(tuple(err_list)).replace(',)', ')')))

        conn.commit()

    except Exception as e:
        logger.error(e)
        logger.info("Trying to reconnect to the DB...")
        conn.close()
        connect_to_db()

    return len(records)


def create_all():
    while True:
        records_len = create_top_n_vectors()
        print(str(records_len) + " vectors created successfully")
        logger.info(str(records_len) + " vectors created successfully")
        time.sleep(GAP)


if __name__ == "__main__":
    connect_to_db()
    create_all()
