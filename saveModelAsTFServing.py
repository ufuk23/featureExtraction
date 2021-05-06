from classification_models.keras import Classifiers
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model


model1 = ResNet50(weights='weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
model = Model(inputs=model1.input, outputs=model1.layers[-2].output)
model.summary()
model.save(filepath='resnet50/1', save_format='tf')


#ResNet18, preprocess_input = Classifiers.get('resnet18')
#model = ResNet18((224, 224, 3), weights='weights/resnet18_imagenet_1000.h5')
#model.summary()
#model.save(filepath='resnet18/1', save_format='tf')
