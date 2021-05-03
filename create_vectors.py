import pyodbc
from tensorflow.keras.preprocessing import image
import requests
import numpy as np
from scipy import spatial
from sklearn.preprocessing import minmax_scale
import json
import time


conn = None
GAP = 60  # seconds to sleep between the loop steps

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
    except Exception as e:
        print(e)


def create_top_n_vectors():
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 100 F.ESER_ID, F.FOTOGRAF_PATH FROM ESER_FOTOGRAF F "
                   "LEFT JOIN ESER E ON F.ESER_ID = E.ID "
                   "WHERE ANA_FOTOGRAF=1 AND DOLASIM_KOPYASI_PATH is NULL AND "
                   "E.AKTIF=1 AND E.SILINMIS=0 order by F.ESER_ID")

    records = cursor.fetchall()

    for row in records:
        print("id: ", row[0], row[1])
        print("\n")
        try:
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
            #print(result_vec)

            # save the vector to the Milvus DB
            ids = []
            vectors = []
            ids.append(str(row[0]))
            vectors.append(result_vec.tolist())
            data_milvus = json.dumps({"ids": ids, "vectors": vectors})
            resp_milvus = requests.post(milvus_url, data=data_milvus, headers=headers)
            print(resp_milvus)

            conn.execute("UPDATE ESER_FOTOGRAF set DOLASIM_KOPYASI_PATH='1' where ANA_FOTOGRAF=1 AND ESER_ID=" + str(row[0]));

        except (FileNotFoundError, IOError):
            print("File not found: " + fs + row[1])
            conn.execute("UPDATE ESER_FOTOGRAF set DOLASIM_KOPYASI_PATH='-1' where ANA_FOTOGRAF=1 AND ESER_ID=" + str(row[0]));
        except ValueError:
            print("Decoding JSON has failed")
        except (requests.HTTPError, requests.RequestException) :
            print("HTTP/Request error occurred: " + e)
        except Exception as e:
            print(e)

    # commit for top N selected records
    conn.commit()

    return len(records)


def create_all():
    while True:
        try:
            records_len = create_top_n_vectors()
            print(str(records_len) + " vectors created successfully")
            if records_len == 0:
                print("No record found to get the vector")
            time.sleep(GAP)
        except Exception as e:
            print(e)
        finally:
            print("Trying to reconnect to the DB")
            conn.close()
            connect_to_db()


if __name__ == "__main__":
    connect_to_db()
    create_all()
