from builtins import print

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from scipy import spatial
from keras.models import Model
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import os

#model = ResNet50(weights='imagenet')
model = ResNet50(weights='weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
model.summary()

vectorDict = {}

for fname in os.listdir('images'):
    # process the files under the directory
    img = image.load_img('images/'+fname, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    feature = model.predict(img_data)
    feature_np = np.array(feature)

    # min-max scale the data between 0 and 1
    scaledVec = minmax_scale(feature_np.flatten())
    roundedVec = np.round(scaledVec, 2)
    vectorDict[fname] = roundedVec

np.savetxt('resnet50_vectors.csv', list(vectorDict.values()), delimiter=',', fmt='%1.2f')
np.savetxt('resnet50_vectorNames.csv', list(vectorDict.keys()), fmt='%s')


input = vectorDict.get('coin1.jpg')
print(input)
print(input.shape)

dResult = {}
for key, value in vectorDict.items():
    cosine_similarity = 1 - spatial.distance.cosine(input, value)
    dResult[key] = cosine_similarity

dSorted = sorted(dResult.items(), key=lambda x: x[1], reverse=True)

for i in dSorted:
    print (i)