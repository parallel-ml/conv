from keras.layers import Conv2D, Input, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K
import numpy as np


def split_xy(X, kernal, stride):
    """ return a list of 3D tensor split by x and y """
    w, h = kernal
    ws, hs = stride
    _, W, H, _ = K.int_shape(X)
    wb = ((W / ws) / 2) * ws + w + 1
    hb = ((H / hs) / 2) * hs + h + 1
    return Lambda(
        lambda x:
        [x[:, :wb, :hb, :], x[:, wb - (w - ws):, :hb, :], x[:, :wb, hb - (h - hs):, :], x[:, wb - (w - ws):, hb - (h - hs):, :]]
    )(X)


def merge(tensors):
    up = Concatenate(axis=1)(tensors[:2])
    down = Concatenate(axis=1)(tensors[2:])
    return Concatenate(axis=2)([up, down])


def conv(tensors, filters, kernal, stride, padding):
    return [Conv2D(filters, kernal, strides=stride, padding=padding)(x) for x in tensors]


def forward(data, filters, kernal, stride=(1, 1), padding='valid'):
    X = Input(data.shape)
    output = merge(conv(split_xy(X, kernal, stride), filters, kernal, stride, padding))
    model = Model(X, output)
    return model.predict(np.array([data]))
