#from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import minmax_scale
from PIL import Image
import flask
import io
import os
import json
import requests
import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

app = flask.Flask(__name__)
#UPLOAD_FOLDER = "/home/muesd/similarity/uploads"
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# logger
app.logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('flask.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)



# handle milvus collection
print("start connecting to Milvus")
connections.connect("default", host="localhost", port="19530")
has = utility.has_collection("artifact")
print(f"Does collection art exist in Milvus: {has}")

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="artifact_type", dtype=DataType.INT64),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=2048)
]

schema = CollectionSchema(fields, "artifact")
print("Collection artifact")
artifact = Collection("artifact", schema, consistency_level="Strong")
print("Start loading Milvus collection")
artifact.load()

search_params = {
    "metric_type": "l2",
    "params": {"nprobe": 16},
}


# Model REST API - tf serving - predict service URL
# tf_serving_url = 'http://172.17.0.4:8501/v1/models/resnet50:predict'
tf_serving_url = 'http://localhost:8501/v1/models/resnet50:predict'

headers = {"content-type": "application/json"}

# MILVUS REST API URL (1.0.0)
# milvus_search_url = 'http://172.17.0.3:19121/collections/kam/vectors'
# milvus_search_url = 'http://localhost:19121/collections/kam/vectors'


def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img)
    #img = image.img_to_array(img) # ustteki satir ile ayni islemi yapiyor, onun icin keras kullanmaya gerek yok
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

        if flask.request.method == "POST":
            if flask.request.files.get("image"):

                #file = flask.request.files["image"]
                #path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                #file.save(path)

                # TODO: load_img de resize olmasiyla olmamasi arasinda sonuc vektor fark ediyor
                # test_img = image.load_img(path, target_size=(224,224))
                #test_img = image.load_img(path) #  bu sekilde PIL imdage load ile ayni oluyor
                print("POST req is done")
                test_img = flask.request.files["image"].read()
                test_img = Image.open(io.BytesIO(test_img))
                # fileName = flask.request.args.get("filename")

        req_artifact_type = flask.request.form.get("artifact_type")
        print("artifactType: ", req_artifact_type)
        req_ids = flask.request.form.get("ids")
        print("img: ", test_img)
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

        # milvus 2.x
        req_expr = "pk in [" + req_ids.strip() +  "]" if req_ids != None else ""
        req_expr = req_expr + (" and " if req_ids != None and req_artifact_type != None else "")
        req_expr = req_expr + (("artifact_type == " + req_artifact_type) if req_artifact_type != None else "")

        resp_milvus = artifact.search(vectors, "embeddings", search_params, limit=100, expr=req_expr, output_fields=["artifact_type"])
        print("req_exp: ", req_expr)
        print("MILVUS RESPONSE:", resp_milvus)
        json_data=[]

        for hits in resp_milvus:
            for h in hits:
                json_data.append({'id': h.id, 'distance':h.distance, 'artifact_type':h.entity.get('artifact_type')})

        print(json.dumps(json_data))
        return json.dumps(json_data)

    except ValueError as e:
        app.logger.error("Decoding JSON has failed")
        app.logger.error(e)
    except (requests.HTTPError, requests.RequestException) as e:
        app.logger.error("HTTP/Request error occurred")
        app.logger.error(e)

    return "{'error': 'Unexpected Error'}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
