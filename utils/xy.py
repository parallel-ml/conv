from keras.layers import Conv2D, Input, Lambda, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K
import numpy as np
import math


def split_xy_2(X, kernal, strides):
    """
        Return a list of 3D tensor split by x and y

        Arithmetic, calculation used:
            output_size = floor((input_size - kernal_size + 2 * padding) / stride) + 1
            left_part_size = output_size / 2
            right_part_size = output_size - left_part_size
            left_input_size = (left_part_size - 1) * stride + kernal_size
            right_input_size = input_size - ...
    """
    wk, hk = kernal
    ws, hs = strides
    _, W, H, _ = K.int_shape(X)

    # calculate boundary
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
    # assemble all tensors on same row
    up, down = Concatenate(axis=1)(tensors[:2]), Concatenate(axis=1)(tensors[2:])
    return Concatenate(axis=2)([up, down])


def split_xy(X, kernal, strides, padding, num):
    """ A general function for split tensors with different shapes. """
    # take care of padding here and set padding of conv always to be valid
    if padding == 'same':
        wk, hk = kernal
        ws, hs = strides
        _, W, H, _ = K.int_shape(X)
        ow, oh = W / ws, H / hs
        if W % ws != 0:
            ow += 1
        if H % hs != 0:
            oh += 1
        wp, hp = (ow - 1) * ws + wk - W, (oh - 1) * hs + hk - H
        wp, hp = wp if wp >= 0 else 0, hp if hp >= 0 else 0
        X = ZeroPadding2D(padding=((hp / 2, hp - hp / 2), (wp / 2, wp - wp / 2)))(X)

    wk, hk = kernal
    ws, hs = strides
    _, W, H, _ = K.int_shape(X)

    # output size
    ow, oh = (W - wk) / ws + 1, (H - hk) / hs + 1

    # calculate boundary for general chunk
    wchunk, hchunk = ow / num, oh / num
    rw, rh = (wchunk - 1) * ws + wk, (hchunk - 1) * hs + hk

    # calculate special boundary for last chunk
    wlchunk, hlchunk = ow - (num - 1) * wchunk, oh - (num - 1) * hchunk
    lrw, lrh = (wlchunk - 1) * ws + wk, (hlchunk - 1) * hs + hk

    offset = lambda kernals, strides, i: (kernals - strides) * i if kernals - strides > 0 else 0

    # create a list of tuple with boundary (left, right, up, down)
    boundary = []
    for r in range(num):
        for c in range(num):
            if r == num - 1 and c == num - 1:
                boundary.append((W - lrw, W, H - lrh, H))
            elif r == num - 1:
                boundary.append((rw * c - offset(wk, ws, c), rw * c - offset(wk, ws, c) + rw, H - lrh, H))
            elif c == num - 1:
                boundary.append((W - lrw, W, rh * r - offset(hk, hs, r), rh * r - offset(hk, hs, r) + rh))
            else:
                boundary.append(
                    (
                        rw * c - offset(wk, ws, c),
                        rw * c - offset(wk, ws, c) + rw,
                        rh * r - offset(hk, hs, r),
                        rh * r - offset(hk, hs, r) + rh,
                    )
                )

    return Lambda(
        lambda x:
        [x[:, lb:rb, ub:db, :] for lb, rb, ub, db in boundary]
    )(X)


def merge(tensors):
    size = int(math.sqrt(len(tensors)))
    rows = [Concatenate(axis=1)(tensors[k * size:k * size + size]) for k in range(size)]
    return Concatenate(axis=2)(rows)


def conv(tensors, filters, kernal, strides, padding, activation):
    layer = Conv2D(filters, kernal, strides=strides, padding=padding, activation=activation)
    return [layer(x) for x in tensors]


def forward(data, filters, kernal, strides=(1, 1), padding='valid'):
    X = Input(data.shape)
    output = merge(conv(split_xy(X, kernal, strides, padding, 3), filters, kernal, strides, 'valid'))
    model = Model(X, output)
    return model.predict(np.array([data]))
