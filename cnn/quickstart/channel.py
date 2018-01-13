from keras.layers import Conv2D, Input, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
import numpy as np


def split(X, filters):
    """ return a list of 3D tensor split by channel """
    return Lambda(lambda x: [x for _ in range(filters)])(X)


def merge(tensors):
    return Concatenate()(tensors)


def conv(tensors, kernal, stride, padding):
    return [Conv2D(1, kernal, strides=stride, padding=padding)(x) for x in tensors]


def forward(data, filters, kernal, stride=(1, 1), padding='valid'):
    X = Input(data.shape)
    output = merge(conv(split(X, filters), kernal, stride, padding))
    model = Model(X, output)
    return model.predict(np.array([data]))

