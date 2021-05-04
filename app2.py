from classification_models.keras import Classifiers
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
from scipy import spatial
from sklearn.preprocessing import minmax_scale
from PIL import Image
import flask
import io
import json

app = flask.Flask(__name__)
model = None
preprocess_input = None
featureDict = {}

def loadModel():
    global model
    global preprocess_input
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model = ResNet18((224, 224, 3), weights='weights/resnet18_imagenet_1000.h5')
    model.summary()
    model.save(filepath='savedModel/similarityModel', save_format='tf')

def testContainer():
    global preprocess_input
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    img = image.load_img("images/coin3.jpg", target_size=(224, 224))
    img_data = prepare_image(img)
    url = 'http://localhost:8501/v1/models/similarityModel:predict'
    headers = {"content-type": "application/json"}
    data = json.dumps({"signature_name": "serving_default", "instances": img_data.tolist()})

    response = requests.post(url, data = data, headers=headers)
    dict = json.loads(response.text)
    feature_np = np.array(dict["predictions"])

    scaledVec = minmax_scale(feature_np.flatten())
    roundedVec = np.round(scaledVec, 2)
    print(roundedVec)

def load_image_featureVectors(path):
    global model
    for i in sorted(os.listdir(path)):
        name = os.path.splitext(i)[0]
        img_path = os.path.join(path, i)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = prepare_image(img)

        featureVector = model.predict(img_data)
        feature_np = np.array(featureVector)

        # min-max scale the data between 0 and 1
        scaledVec = minmax_scale(feature_np.flatten())
        roundedVec = np.round(scaledVec, 2)
        # featureDict[name] = roundedVec

    print ("image vectors loaded: " , len(featureDict))

def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


### test POST method with ----> curl -X POST -F image=@images/coin.jpeg 'http://localhost:5000/predict'
### test GET method with browser ---> http://localhost:5000/predict?image=coin.jpeg

@app.route("/predict", methods=["GET", "POST"])
def predict():
    test_img = None
    fileName = None
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            test_img = flask.request.files["image"].read()
            test_img = Image.open(io.BytesIO(test_img))
            fileName = flask.request.args.get("f")

    elif flask.request.method == "GET":
        imgName = flask.request.args.get("image")
        test_img = image.load_img("images/" + imgName, target_size=(224, 224))

    input = prepare_image(test_img)

    testVector = model.predict(input)

    testVector_np = np.array(testVector)
    # min-max scale the data between 0 and 1
    scaledVec = minmax_scale(testVector_np.flatten())
    testVec = np.round(scaledVec, 2)

    featureDict[fileName] = testVec

    result_dict = {}
    for name in featureDict:
        cosineDistance = 1 - spatial.distance.cosine(testVec, featureDict[name])
        result_dict[name] = cosineDistance

    dSorted = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    for i in dSorted:
        print(i)

    data = {"success": False}
    data["predictions"] = []

    for k in sorted(result_dict, key=(lambda k: result_dict[k]), reverse=True):
        r = {"label": k, "probability":result_dict[k]}
        data["predictions"].append(r)
    data["success"] = True
    return flask.jsonify(data)

if __name__ == "__main__":
    loadModel()
    load_image_featureVectors("images")
    testContainer()
    app.run(host='0.0.0.0', port=5000)
