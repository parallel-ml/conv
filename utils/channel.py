"""
    This module provides single Conv2D layer channel wise split.
    Technique used is simple that divide filter into number of
    batches and copy identical inputs onto batch of filters.
"""
from keras.layers import Conv2D, Input, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
import numpy as np


def split(X, num):
    """ Return a list of 3D tensor split by channel. """
    return Lambda(lambda x: [x for _ in range(num)])(X)


def merge(tensors):
    return Concatenate()(tensors)


def conv(tensors, filters, kernal, strides, padding, activation):
    size = []
    for _ in range(len(tensors) - 1):
        size.append(filters / len(tensors))
    size.append(filters - filters / len(tensors) * (len(tensors) - 1))

    return [Conv2D(size[i], kernal, strides=strides, padding=padding, activation=activation)(x) for
            i, x in enumerate(tensors)]


def forward(data, filters, kernal, strides=(1, 1), padding='valid'):
    X = Input(data.shape)
    output = merge(conv(split(X, 3), filters, kernal, strides, padding, 'relu'))
    model = Model(X, output)
    return model.predict(np.array([data]))
