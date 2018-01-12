from keras.layers import Conv2D, Input, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K
import numpy as np


def split(X):
    """ return a list of 3D tensor split by channel """
    return Lambda(lambda x: [x[:, :, :, k:k + 1] for k in range(K.int_shape(x)[-1])])(X)


def merge(tensors):
    return Concatenate(axis=-1)(tensors)


def conv(tensors, filters, kernal, stride, padding):
    return [Conv2D(filters=filters, kernel_size=kernal, strides=stride, padding=padding)(x) for x in tensors]


def forward(data, filters, kernal, stride=(1, 1), padding='same'):
    X = Input(data.shape)
    output = merge(conv(split(X), filters, kernal, stride, padding))
    model = Model(X, output)
    return model.predict(np.array([data]))

