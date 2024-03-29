# for keras
from classification_models.keras import Classifiers
from tensorflow.keras.preprocessing import image
from keras.models import Model
import numpy as np
import os
from scipy import spatial
from sklearn.preprocessing import minmax_scale
from numpy import savetxt

ResNet18, preprocess_input = Classifiers.get('resnet18')
#model = ResNet18((224, 224, 3), weights='imagenet')
model = ResNet18((224, 224, 3), weights='weights/resnet18_imagenet_1000.h5')
model.summary()

# for output dim 512 (pops last 2 layers)
# model1 = ResNet18((224, 224, 3), weights='weights/resnet18_imagenet_1000.h5')
# model= Model(inputs=model1.input, outputs=model1.layers[-3].output)

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

np.savetxt('resnet18_vectors.csv', list(vectorDict.values()), delimiter=',', fmt='%1.2f')
np.savetxt('resnet18_vectorNames.csv', list(vectorDict.keys()), fmt='%s')


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