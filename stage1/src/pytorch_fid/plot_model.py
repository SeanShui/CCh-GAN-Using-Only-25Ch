from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import numpy as np
import visualkeras

#model = ResNet50(weights='imagenet')
#model = InceptionV3(include_top=False, pooling='avg', input_shape=(224,224,3))
model = VGG16(weights='imagenet', input_shape=(224,224,3))

plot_model(model, to_file='model.png')

#visualkeras.layered_view(model)
visualkeras.layered_view(model, to_file='output.png').show()

