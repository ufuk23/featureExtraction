from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import minmax_scale
from scipy import spatial
import os
import numpy as np

#model = VGG16(weights='imagenet', include_top=False)
model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
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

    # mix-max scale the data between 0 and 1
    feature_scaled = minmax_scale(feature_np.flatten())
    vectorDict[fname] = np.round(feature_scaled, 2)

input = vectorDict.get('elon1.jpg')
print(input)

dResult = {}
for key, value in vectorDict.items():
    cosine_similarity = 1 - spatial.distance.cosine(input, value)
    dResult[key] = cosine_similarity

dSorted = sorted(dResult.items(), key=lambda x: x[1], reverse=True)

for i in dSorted:
    print (i)