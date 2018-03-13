from keras.layers import Conv2D, Input, Lambda, SeparableConv2D
from keras.layers.merge import Concatenate
from keras.models import Model
import numpy as np


def split(X, num):
    """ Return a list of 3D tensor split by channel. """
    return Lambda(lambda x: [x for _ in range(num)])(X)


def merge(tensors):
    return Concatenate()(tensors)


def conv(tensors, filters, kernal, strides, padding, activation, separable, use_bias):
    size = []
    for _ in range(len(tensors) - 1):
        size.append(filters / len(tensors))
    size.append(filters - filters / len(tensors) * (len(tensors) - 1))

    if separable:
        return [SeparableConv2D(size[i], kernal, strides=strides, padding=padding, activation=activation,
                                use_bias=use_bias)(x) for i, x in enumerate(tensors)]

    return [Conv2D(size[i], kernal, strides=strides, padding=padding, activation=activation, use_bias=use_bias)(x) for
            i, x in enumerate(tensors)]


def forward(data, filters, kernal, strides=(1, 1), padding='valid'):
    X = Input(data.shape)
    output = merge(conv(split(X, 3), filters, kernal, strides, padding, 'same', 'relu'))
    model = Model(X, output)
    return model.predict(np.array([data]))
