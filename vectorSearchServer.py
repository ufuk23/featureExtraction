from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import minmax_scale
from PIL import Image
import flask
import io
import os
import json
import requests
import logging


app = flask.Flask(__name__)
UPLOAD_FOLDER = "/home/muesd/similarity/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# logger
app.logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('flask.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)

# Model REST API - tf serving - predict service URL
tf_serving_url = 'http://localhost:8501/v1/models/similarityModel:predict'
headers = {"content-type": "application/json"}

# MILVUS REST API URL
milvus_search_url = 'http://localhost:19121/collections/artifact/vectors'


def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    return img


# gets the feature vector from tf serving model and search it on Milvus DB
@app.route("/predict", methods=["POST"])
def predict():
    try:
        test_img = None
        vectors = []

        app.logger.info("prediction service has called")
        print("prediction service has called")

        if flask.request.method == "POST":
            if flask.request.files.get("image"):

                #file = flask.request.files["image"]
                #path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                #file.save(path)

                # TODO: load_img de resize olmasiyla olmamasi arasinda sonuc vektor fark ediyor
                # test_img = image.load_img(path, target_size=(224,224))
                #test_img = image.load_img(path) #  bu sekilde PIL imdage load ile ayni oluyor

                test_img = flask.request.files["image"].read()
                test_img = Image.open(io.BytesIO(test_img))
                # fileName = flask.request.args.get("filename")

        print(test_img)
        img_data = prepare_image(test_img)

        # prepare for tf serving service
        # give the photo and get the vector from the model
        data = json.dumps({"signature_name": "serving_default", "instances": img_data.tolist()})
        response = requests.post(tf_serving_url, data=data, headers=headers)
        dict_resp = json.loads(response.text)
        feature_np = np.array(dict_resp["predictions"])
        # min-max scale the data between 0 and 1
        scaled_vec = minmax_scale(feature_np.flatten())
        result_vec = np.round(scaled_vec, 2)
        vectors.append(result_vec.tolist())
        print(result_vec)

        json_milvus = {
            'search': {
                'topk': 10,
                'vectors': vectors,
                'params': {
                    'nprobe': 16
                }
            }
        }

        resp_milvus = requests.put(milvus_search_url, data=json.dumps(json_milvus), headers=headers) 
        # app.logger.info("rep_milvus")
        print(resp_milvus.text)
        return resp_milvus.text

    except ValueError as e:
        app.logger.error("Decoding JSON has failed")
        app.logger.error(e)
    except (requests.HTTPError, requests.RequestException) as e:
        app.logger.error("HTTP/Request error occurred")
        app.logger.error(e)

    return "{'error': 'Unexpected Error'}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
