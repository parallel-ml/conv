from keras.layers.convolutional import Conv2D, MaxPooling2D
from ..utils import channel as conv_channel
from ..utils import xy as conv_xy
from ..utils import filter as conv_filter


def conv_unit(X, nb_filter, kernal, activation='relu', max_pooling=True, stride=(1, 1)):
    if max_pooling:
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)
    X = Conv2D(nb_filter, kernel_size=kernal, activation=activation, strides=stride, padding='same')(X)
    return X


def xy_unit(X, filters, kernal, max_pooling=True, stride=(1, 1), num=3, padding='same', activation='relu'):
    """ Cnn unit with spatial separation. """
    if max_pooling:
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)
    X = conv_xy.split_xy(X, kernal, stride, padding, num)
    # keep padding of conv2d layer always to be valid
    X = conv_xy.conv(X, filters, kernal, stride, 'valid', activation)
    X = conv_xy.merge(X)
    return X


def channel_unit(X, filters, kernal, max_pooling=True, stride=(1, 1), padding='same', activation='relu'):
    """ Cnn unit with channel separation. """
    if max_pooling:
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)
    X = conv_channel.split(X, 3)
    X = conv_channel.conv(X, filters, kernal, stride, padding, activation)
    X = conv_channel.merge(X)
    return X


def filter_unit(X, filters, kernal, max_pooling=True, stride=(1, 1), padding='same', activation='relu'):
    """ Cnn unit with depth wise separation. """
    if max_pooling:
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)
    X = conv_filter.split(X, 3)
    X = conv_filter.conv(X, filters, kernal, stride, padding, activation)
    X = conv_filter.merge(X)
    return X
