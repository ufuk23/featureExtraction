from scipy import spatial
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.backend import set_session
from classification_models.keras import Classifiers
from sklearn.preprocessing import minmax_scale
import os
import io
import numpy as np
import flask
import tensorflow as tf
from PIL import Image

app = flask.Flask(__name__)
graph = None
session = None
model = None
featureDict = {}

def loadModel():
    global model
    global graph
    global session
    graph = tf.get_default_graph()
    session = tf.Session()
    set_session(session)
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model = ResNet18((224, 224, 3), weights='weights/resnet18_imagenet_1000.h5')
    model.summary()

def load_image_featureVectors(path):
    global model
    for i in sorted(os.listdir(path)):
        name = os.path.splitext(i)[0]
        img_path = os.path.join(path, i)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        featureVector = model.predict(img_data)
        feature_np = np.array(featureVector)

        # min-max scale the data between 0 and 1
        scaledVec = minmax_scale(feature_np.flatten())
        roundedVec = np.round(scaledVec, 2)
        featureDict[name] = roundedVec

    print ("image vectors loaded: " , len(featureDict))

def prepare_image(image, target=(224,224)):
    if image.mode != "RGB":
        image = image.convert("RGB")


### test POST method with ----> curl -X POST -F image=@images/coin.jpeg 'http://localhost:5000/predict'
### test GET method with browser ---> http://localhost:5000/predict?image=coin.jpeg

@app.route("/predict", methods=["GET", "POST"])
def predict():
    global graph
    global session
    with graph.as_default():
        set_session(session)

        test_img = None
        if flask.request.method == "POST":
            if flask.request.files.get("image"):
                test_img = flask.request.files["image"].read()
                test_img = Image.open(io.BytesIO(test_img))
                test_img = test_img.resize((224, 224), Image.ANTIALIAS)

        elif flask.request.method == "GET":
            imgName = flask.request.args.get("image")
            test_img = image.load_img("images/" + imgName, target_size=(224, 224))

        input = image.img_to_array(test_img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)

        testVector = model.predict(input)

        result_dict = {}
        for name in featureDict:
            cosineDistance = 1 - spatial.distance.cosine(testVector, featureDict[name])
            print(name, "CosineDist=", cosineDistance)
            result_dict[name] = cosineDistance

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
    app.run(host='0.0.0.0', port=5000)