# for keras
from classification_models.keras import Classifiers
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from scipy import stats

a = np.array([1.0e-5,3.0e-6,1.0e-4])
normalized_a = np.round(a/np.linalg.norm(a,1.0),4)
# [0.2,0.6,0.2]
print(normalized_a)

data = np.array([[1000, 10, 0.5],
    [765, 5, 0.35],
    [8e-8, 7, 0.09], ])
data = normalize(data, axis=0, norm='max')
print(data)

data = np.array([1.18152776e-02, 1.28207034e-07, 9.01502961e-09, 3.83313659e-11,
 4.53214213e-08, 2.88190938e-09, 1.91198413e-09, 9.43679410e-08])

feature_scaled = minmax_scale(data.flatten())
print(feature_scaled)

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18((224, 224, 3), weights='weights/resnet18_imagenet_1000.h5')

img = image.load_img('train/1.jpg', target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

feature = model.predict(img_data)

feature_np = np.array(feature)
input = feature_np.flatten()
print(input)

print('----------------------------------------------------------------------------')

normalized_a = np.round(input/np.linalg.norm(input,1.0),4)

#normal_array = normalize(input.reshape(1, -1), axis=0, norm='max')
#normal_array = input / input.max(axis=0)
normal_array = minmax_scale(input)

boxcox = StandardScaler().fit_transform(feature_np);

print(normal_array)
#normal_array = normal_array * 1e6;

rounded = np.round(normal_array, 4)

data = normalize(feature_np, axis=0, norm='max')
print (data)