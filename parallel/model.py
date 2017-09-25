import gc
import sys
from sys import argv
import time
import cv2
import keras
import numpy as np
from keras import backend as K
from memory_profiler import profile

image = cv2.imread('data/tiger.jpg')
resized_image = cv2.resize(image, (224, 224))

start = time.time()

class Network:
    def __init__(self, type='resnet'):
        if type == 'resnet':
            self.NN = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None,
                                                           input_shape=(224, 224, 3), pooling=None, classes=1000)
        elif type == 'vgg19':
            self.NN = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None,
                                                     input_shape=(224, 224, 3), pooling=None, classes=1000)
        elif type == 'mobilenet':
            self.NN = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1,
                                                             dropout=1e-3, include_top=True, weights='imagenet',
                                                             input_tensor=None, pooling=None, classes=1000)
        else:
            sys.exit('resnet, vgg19, mobilenet')

    def predict(self, image):
        """
        :param image: expected input is numpy array of images
        :return:
        """
        return self.NN.predict(image)

    # try to clean up memory used by keras model, but it seems the framework cannot do that
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.NN
        del self
        K.clear_session()
        gc.collect()


def timestamp(output=''):
    global start
    if output:
        print '{}: {:.2f}s'.format(output, time.time() - start)
    start = time.time()


def log(output=''):
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print 'running network: ', output


@profile
def run_network(ntype):
    log(ntype)
    timestamp()
    # keras provides API for directly geting a 50-layer res-net model with imagenet weight
    with Network(ntype) as NN:
        timestamp('load model')
        # the input passed into keras forwarding function is expected to be 4D
        # the image is 3D and the input is a list of different images
        test_x = np.array([resized_image])
        # output probability of being in each of 1000 classes
        timestamp()
        NN.predict(test_x)
        timestamp('inference')


if __name__ == '__main__':
    if len(argv) < 2:
        sys.exit('python quickstart.py [type of your model]')
    ntype = argv[1]
    run_network(ntype)
