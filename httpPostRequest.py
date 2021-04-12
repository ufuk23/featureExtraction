#from classification_models.keras import Classifiers
#from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests
from scipy import spatial
from sklearn.preprocessing import minmax_scale
import json
from PIL import Image

preprocess_input = None

def load_image_featureVectors(path):
    #global preprocess_input
    #ResNet18, preprocess_input = Classifiers.get('resnet18')

    img = image.load_img("images/coin3.jpg", target_size=(224, 224))
    #img = Image.open("images/coin3.jpg")
    img_data = prepare_image(img)

    #featureVector = model.predict(img_data)
    url = 'http://localhost:8501/v1/models/similarityModel:predict'
    headers = {"content-type": "application/json"}
    #print(img_data)
    #print(img_data.shape)

    data = json.dumps({"signature_name": "serving_default", "instances": img_data.tolist()})
    #print(json_str)

    response = requests.post(url, data = data, headers=headers)
    dict = json.loads(response.text)
    #print(dict["predictions"])
    feature_np = np.array(dict["predictions"])

    # min-max scale the data between 0 and 1
    scaledVec = minmax_scale(feature_np.flatten())
    roundedVec = np.round(scaledVec, 2)
    print(roundedVec)

def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    img = image.img_to_array(img)
    #img = np.array(img)
    img = np.expand_dims(img, axis=0)
    #img = preprocess_input(img)
    return img

if __name__ == "__main__":
    load_image_featureVectors("images")
