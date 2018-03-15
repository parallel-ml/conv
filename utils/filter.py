"""
    This module implements single Conv2D layer filter wise split.
    Compared to channel wise split, instead of splitting the filter
    parameter, it keeps the same filter but splits the input. At the
    end, it adds everything back.
"""
from keras.layers import Conv2D, Input, Lambda, Activation
from keras.layers.merge import Add
from keras.models import Model
import keras.backend as K
import numpy as np


def split(X, num):
    depth = K.int_shape(X)[-1]
    d, dl = depth / num, depth - (num - 1) * (depth / num)
    boundary = []
    for i in range(num):
        if i != num - 1:
            boundary.append((i * d, (i + 1) * d))
        else:
            boundary.append((depth - dl, depth))
    return Lambda(lambda x: [x[:, :, :, lb:rb] for lb, rb in boundary])(X)


def merge(tensors, activation):
    X = Add()([x for x in tensors])
    if activation is not None:
        X = Activation(activation)(X)
    return X


def conv(tensors, filters, kernal, strides, padding):
    return [Conv2D(filters, kernal, strides=strides, padding=padding)(x) for x
            in tensors]


def forward(data, filters, kernal, strides=(1, 1), padding='valid'):
    X = Input(data.shape)
    output = merge(conv(split(X, 4), filters, kernal, strides, padding))
    model = Model(X, output)
    return model.predict(np.array([data]))
