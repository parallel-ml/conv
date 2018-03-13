from keras.layers import Flatten, Dense, Input
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from lrn import LRN2D  # use only during training
from unit import filter_unit, xy_unit, channel_unit, conv_unit


def filter():
    img = Input(shape=(220, 220, 3))

    x = filter_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), padding='same', activation='relu')
    x = filter_unit(x, 128, (5, 5), padding='same', activation='relu')

    x = filter_unit(x, 192, (3, 3), padding='same', activation='relu')
    x = filter_unit(x, 192, (3, 3), max_pooling=False, padding='same', activation='relu')
    x = filter_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)


def xy():
    img = Input(shape=(220, 220, 3))

    x = xy_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), padding='same', activation='relu')
    x = xy_unit(x, 128, (5, 5), padding='same', activation='relu')

    x = xy_unit(x, 192, (3, 3), padding='same', activation='relu')
    x = xy_unit(x, 192, (3, 3), max_pooling=False, padding='same', activation='relu')
    x = xy_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)


def channel():
    img = Input(shape=(220, 220, 3))

    x = channel_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), padding='same', activation='relu')
    x = channel_unit(x, 128, (5, 5), padding='same', activation='relu')

    x = channel_unit(x, 192, (3, 3), padding='same', activation='relu')
    x = channel_unit(x, 192, (3, 3), max_pooling=False, padding='same', activation='relu')
    x = channel_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)


def original():
    img = Input(shape=(220, 220, 3))

    x = conv_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), activation='relu')
    x = conv_unit(x, 128, (5, 5), activation='relu')

    x = conv_unit(x, 192, (3, 3), activation='relu')
    x = conv_unit(x, 192, (3, 3), max_pooling=False, activation='relu')
    x = conv_unit(x, 128, (3, 3), max_pooling=False, activation='relu')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)
