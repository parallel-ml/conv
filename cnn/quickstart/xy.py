from keras.layers import Conv2D, Input, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K
import numpy as np


def split_xy_2(X, kernal, stride):
    """
        return a list of 3D tensor split by x and y

        output_size = floor((input_size - kernal_size + 2 * padding) / stride) + 1
        left_part_size = output_size / 2
        right_part_size = output_size - left_part_size
        left_input_size = (left_part_size - 1) * stride + kernal_size
        right_input_size = input_size - ...
    """
    wk, hk = kernal
    ws, hs = stride
    _, W, H, _ = K.int_shape(X)
    ow, oh = (W - wk) / ws + 1, (H - hk) / hs + 1
    wl, wr = ow / 2, ow - ow / 2
    wlb, wrb = (wl - 1) * ws + wk, W - ((wr - 1) * ws + wk)
    hl, hr = oh / 2, oh - oh / 2
    hlb, hrb = (hl - 1) * hs + hk, H - ((hr - 1) * hs + hk)
    return Lambda(
        lambda x:
        [x[:, :wlb, :hlb, :], x[:, wrb:, :hlb, :], x[:, :wlb, hrb:, :], x[:, wrb:, hrb:, :]]
    )(X)


def merge_2(tensors):
    up = Concatenate(axis=1)(tensors[:2])
    down = Concatenate(axis=1)(tensors[2:])
    return Concatenate(axis=2)([up, down])


def conv(tensors, filters, kernal, stride, padding):
    return [Conv2D(filters, kernal, strides=stride, padding=padding)(x) for x in tensors]


def forward(data, filters, kernal, stride=(1, 1), padding='valid'):
    X = Input(data.shape)
    output = merge_2(conv(split_xy_2(X, kernal, stride), filters, kernal, stride, padding))
    model = Model(X, output)
    return model.predict(np.array([data]))
