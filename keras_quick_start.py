import cv2
import keras
import numpy as np

image = cv2.imread('data/tiger.jpg')
# for res-net the expected input image size is 224*224*3
resized_image = cv2.resize(image, (224, 224))
# keras provides API for directly geting a 50-layer res-net model with imagenet weight
NN = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None,
                                          input_shape=(224, 224, 3), pooling=None, classes=1000)
# the input passed into keras forwarding function is expected to be 4D
# the image is 3D and the input is a list of different images
test_x = np.array([resized_image])
# output probability of being in each of 1000 classes
test_y = NN.predict(test_x)
