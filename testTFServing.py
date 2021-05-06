from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from scipy import spatial
from keras.models import Model
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import json
import requests


def prepare_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

img = image.load_img("images/coin3.jpg", target_size=(224, 224))
img_data = prepare_image(img)
url = 'http://localhost:8501/v1/models/resnet50:predict'
headers = {"content-type": "application/json"}
data = json.dumps({"signature_name": "serving_default", "instances": img_data.tolist()})

response = requests.post(url, data = data, headers=headers)
dict = json.loads(response.text)
feature_np = np.array(dict["predictions"])

scaledVec = minmax_scale(feature_np.flatten())
roundedVec = np.round(scaledVec, 2)
print(roundedVec)
