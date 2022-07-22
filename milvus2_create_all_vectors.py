import pymssql
from tensorflow.keras.preprocessing import image
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
file_handler = logging.FileHandler('log_KAM_create_vector.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add file handler to logger
logger.addHandler(file_handler)

conn = None
GAP = 10  # seconds to sleep between the loop steps

# handle milvus collection
print("start connecting to Milvus")
connections.connect("default", host="localhost", port="19530")
has = utility.has_collection("artifact")
print(f"Does collection artifact exist in Milvus: {has}")

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="artifact_type", dtype=DataType.INT64),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=2048)
]

# description as "artifact" below is mandatory to connect
schema = CollectionSchema(fields, "artifact")
print("Collection artifact")
artifact = Collection("artifact", schema, consistency_level="Strong")


# Model REST API - tf serving - predict service URL
# tf_serving_url = 'http://localhost:8501/v1/models/similarityModel:predict'
tf_serving_url = 'http://localhost:8501/v1/models/resnet50:predict'
headers = {"content-type": "application/json"}

# mount path to access the file Server
# fs = "/mnt/muesfs/mues-images/image/ak/" # prod
fs_kam = "/mnt/muesfs/mues/mueskam-images/dev/image/ak/" # dev
fs_mues = "/mnt/muesfs/mues/mues-images/dev/image/ak/"

# MILVUS REST API URL
#milvus_url = 'http://localhost:19121/collections/artifact/vectors'
#milvus_url = 'http://localhost:19121/collections/kam/vectors'


def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = image.img_to_array(img)
    # img = np.array(img)
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



def create_MUES_vectors():
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

            img = Image.open(fs_mues + row[1])
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
            artifact_types.append(1)
            # print(artifact_types)
            vectors.append(result_vec.tolist())

            ok_list.append(str(row[0]))

        except (FileNotFoundError, IOError):
            logger.error("File not found: " + fs_mues + row[1])
            err_list.append(str(row[0]))  # marking for FileNotFound
        except ValueError as e:
            logger.error("Decoding JSON has failed")
            logger.error(e)
        except (requests.HTTPError, requests.RequestException) as e:
            logger.error("HTTP/Request error occurred")
            logger.error(e)

    try:
        # save the n vector to the Milvus DB
        if(len(vectors) > 0):
            entities = [ids, artifact_types, vectors]
            # print(entities)
            insert_result = artifact.insert(entities)
            # print(f"Number of entities in Milvus: {artifact.num_entities}")  # check the num_entites
    except Exception as e:
        logger.error("MILVUS post request error")
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




def create_KAM_vectors():
    cursor = conn.cursor()
    cursor.execute("select DISTINCT TOP 100 K.uid, F.FOTOGRAF_PATH, F.artifactId from KAM_ARTIFACT_VIEW K "
                   "LEFT JOIN Kam_ArtifactPhotograph F ON K.artifactId = F.artifactId "
                   "WHERE K.artifactType!='INVENTORY_ARTIFACT' AND K.aktif=1 AND K.silinmis=0 AND F.ANA_FOTOGRAF=1 AND F.FOTOGRAF_PATH is not null AND F.FEATURE_VECTOR_STATE is NULL ORDER BY K.uid")

    records = cursor.fetchall()

    ids = []
    vectors = []
    artifact_types = []
    ok_list = []
    err_list = []

    for row in records:
        try:
            logger.info("uid:" + str(row[0]) + " : " + str(row[1]))
            print(("uid: " + str(row[0]) + " : " + str(row[1])))

            img = image.load_img(fs_kam + row[1])
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
            # KAM artifact_type = 2, MUES artifact_type = 1
            artifact_types.append(2)
            # print(artifact_types)
            vectors.append(result_vec.tolist())

            ok_list.append(str(row[2]))

        except (FileNotFoundError, IOError):
            logger.error("KAM - File not found: " + fs_kam + row[1])
            err_list.append(str(row[0]))  # marking for FileNotFound
        except ValueError as e:
            logger.error("KAM - Decoding JSON has failed")
            logger.error(e)
        except (requests.HTTPError, requests.RequestException) as e:
            logger.error("KAM - HTTP/Request error occurred")
            logger.error(e)

    try:
        # save the n vector to the Milvus DB
        if(len(vectors)):
            entities = [ids, artifact_types, vectors]
            insert_result = artifact.insert(entities)
    except Exception as e:
        logger.error("KAM - MILVUS post request error (KAM)")
        logger.error(e)

    try:
        # commit for top N selected records
        if(len(ok_list)>0):
            cursor.execute("UPDATE Kam_ArtifactPhotograph set FEATURE_VECTOR_STATE='1' where ANA_FOTOGRAF=1 AND artifactId in {}".format(str(tuple(ok_list)).replace(',)', ')')))
        if(len(err_list)>0):
            cursor.execute("UPDATE Kam_ArtifactPhotograph set FEATURE_VECTOR_STATE='-1' where ANA_FOTOGRAF=1 AND artifactId in {}".format(str(tuple(err_list)).replace(',)', ')')))

        conn.commit()

    except Exception as e:
        logger.error(e)
        logger.info("KAM - Trying to reconnect to the DB...")
        conn.close()
        connect_to_db()

    return len(records)


def create_all():
    while True:
        records_len = create_MUES_vectors()
        print(str(records_len) + " MUES vectors created successfully")
        logger.info(str(records_len) + " MUES vectors created successfully")
        time.sleep(GAP)

        records_len = create_KAM_vectors()
        print(str(records_len) + " KAM vectors created successfully")
        logger.info(str(records_len) + " KAM vectors created successfully")
        time.sleep(GAP)


if __name__ == "__main__":
    connect_to_db()
    create_all()
