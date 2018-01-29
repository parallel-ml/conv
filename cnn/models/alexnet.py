from keras.layers import Flatten, Dense, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from lrn import LRN2D  # use only during training
from cnn.utils import channel as conv_channel
from cnn.utils import xy as conv_xy


def conv2D_bn(X, nb_filter, kernal, activation='relu', batch_norm=True, max_pooling=True, stride=(1, 1)):
    if max_pooling:
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)
    X = Conv2D(nb_filter, kernel_size=kernal, activation=activation, strides=stride, padding='same')(X)
    return X


def xy_unit(X, filters, kernal, max_pooling=True, stride=(1, 1), num=3):
    """ cnn unit with spatial separation """
    if max_pooling:
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)
    X = conv_xy.split_xy(X, kernal, stride, num)
    X = conv_xy.conv(X, filters, kernal, stride, 'same')
    X = conv_xy.merge(X)
    return X


def channel_unit(X, filters, kernal, max_pooling=True, stride=(1, 1)):
    """ cnn unit with channel separation """
    if max_pooling:
        X = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(X)
    X = conv_channel.split(X, filters)
    X = conv_channel.conv(X, kernal, stride, 'same')
    X = conv_channel.merge(X)
    return X


def filter_unit(X):
    """ cnn unit with depth wise separation """
    pass


def xy():
    img = Input(shape=(220, 220, 3))

    x = xy_unit(img, 48, (11, 11), max_pooling=False, stride=(4, 4))
    x = xy_unit(x, 128, (5, 5))

    x = xy_unit(x, 192, (3, 3))
    x = xy_unit(x, 192, (3, 3), max_pooling=False)
    x = xy_unit(x, 128, (3, 3), max_pooling=False)

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)


def channel():
    img = Input(shape=(220, 220, 3))

    x = channel_unit(img, 48, (11, 11), max_pooling=False, stride=(4, 4))
    x = channel_unit(x, 128, (5, 5))

    x = channel_unit(x, 192, (3, 3))
    x = channel_unit(x, 192, (3, 3), max_pooling=False)
    x = channel_unit(x, 128, (3, 3), max_pooling=False)

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)


def original():
    img = Input(shape=(220, 220, 3))

    x = conv2D_bn(img, 48, (11, 11), max_pooling=False, stride=(4, 4))
    x = conv2D_bn(x, 128, (5, 5))

    x = conv2D_bn(x, 192, (3, 3))
    x = conv2D_bn(x, 192, (3, 3), max_pooling=False)
    x = conv2D_bn(x, 128, (3, 3), max_pooling=False)

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)
