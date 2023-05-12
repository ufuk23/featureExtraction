from configparser import ConfigParser
import pymssql
from PIL import Image
import requests
import numpy as np
from scipy import spatial
from sklearn.preprocessing import minmax_scale
import json
import time
import logging

# Get the configparser object
config_object = ConfigParser(interpolation=None)

# path
config_path = "/muesconfig/conf.ini"

# Read config
config_object.read(config_path)

# Get the database config object
global dbinfo
dbinfo = config_object["DATABASE"]
artifact_type = dbinfo["artifact_type"]

# config logs
print("DATABASE = " + dbinfo["database"])
print("artifact_type = " + dbinfo["artifact_type"])
print("select_query = " + dbinfo["select_query"])


# Gets or creates a logger
logger = logging.getLogger(__name__)
# set log level
logger.setLevel(logging.DEBUG)
# define file handler and set formatter
file_handler = logging.FileHandler('log_store_vector.log')
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
fs = "/image_path/" # mapping volume parameter
print("fileSystem mapped image path = " + fs)

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
        # conn = pymssql.connect(server='10.1.37.177', port='1033', user='muest', password='Mues*test.1', database='mues_test')
        conn = pymssql.connect(server=dbinfo["ip"], port=dbinfo["port"], user=dbinfo["user"], password=dbinfo["password"], database=dbinfo["database"])
        logger.info('DB connected successfully')
    except Exception as e:
        logger.critical(e)


def create_top_n_vectors():
    cursor = conn.cursor()
    cursor.execute(dbinfo["select_query"])

    records = cursor.fetchall()

    sql_for_exception = dbinfo["update_query_failure"]

    for row in records:
        try:
            logger.info("id:" + str(row[0]) + " : " + str(row[1]))
            print(("id: " + str(row[0]) + " : " + str(row[1])))

            # if no saved vector, get vector by tensorflow-serving service
            if row[3] == None:

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

                vector = result_vec.tolist()
                # print(result_vec)

            else:
                vector = json.loads(row[3])
                print("the vector is in the DB")

            # mues uid
            uid = row[0]

            if artifact_type != 1:
                # kam=2, omk=3  to make it unique id
                uid = int(artifact_type) * 100000000 + int(row[0])

            json_data = {
                "points":[
                    {
                        "id": uid,
                        "payload": {"artifact": artifact_type},
                        "vector": vector
                    }
                ]
            }

            data_json = json.dumps(json_data)
            # logger.info(data_json)

            # put it to the qdrant
            response = requests.put(qdrant_url, data=data_json, headers=headers)

            # update state and vector
            params = ('1', json.dumps(vector), row[0])
            cursor.execute(dbinfo["update_query_success"], params)
            conn.commit()
            print("commit performed")

        except (FileNotFoundError, IOError):
            logger.error("File not found: " + fs + row[1])
            cursor.execute(sql_for_exception, row[0])
            conn.commit()
        except ValueError as e:
            logger.error("Decoding JSON has failed")
            cursor.execute(sql_for_exception, row[0])
            conn.commit()
            logger.error(e)
        except (requests.HTTPError, requests.RequestException) as e:
            logger.error("HTTP/Request error occurred")
            cursor.execute(sql_for_exception, row[0])
            conn.commit()
            logger.error(e)
        except Exception as e:
            logger.error(e)
            cursor.execute(sql_for_exception, row[0])
            conn.commit()
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
