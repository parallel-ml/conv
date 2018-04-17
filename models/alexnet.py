from keras.layers import Flatten, Dense, Input
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from unit import filter_unit, xy_unit, channel_unit, conv_unit


NAME = 'alexnet'


def filter(include_fc=True):
    name = NAME + '_filter'
    img = Input(shape=(220, 220, 3))

    x = filter_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), padding='same',
                    activation='relu', name=name + '_1')
    x = filter_unit(x, 128, (5, 5), padding='same', activation='relu', name=name + '_2')

    x = filter_unit(x, 192, (3, 3), padding='same', activation='relu', name=name + '_3')
    x = filter_unit(x, 192, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_4')
    x = filter_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_5')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(4096, activation='relu', name=name + '_6_dense')(fc)
        fc = Dense(4096, activation='relu', name=name + '_7_dense')(fc)
        fc = Dense(1000, activation='softmax', name=name + '_8_dense')(fc)
        x = fc

    return Model(img, x)


def xy(include_fc=True):
    name = NAME + '_spatial'
    img = Input(shape=(220, 220, 3))

    x = xy_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), padding='same',
                activation='relu', name=name + '_1')
    x = xy_unit(x, 128, (5, 5), padding='same', activation='relu', name=name + '_2')

    x = xy_unit(x, 192, (3, 3), padding='same', activation='relu', name=name + '_3')
    x = xy_unit(x, 192, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_4')
    x = xy_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_5')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(4096, activation='relu', name=name + '_6_dense')(fc)
        fc = Dense(4096, activation='relu', name=name + '_7_dense')(fc)
        fc = Dense(1000, activation='softmax', name=name + '_8_dense')(fc)
        x = fc

    return Model(img, x)


def channel(include_fc=True):
    name = NAME + '_channel'
    img = Input(shape=(220, 220, 3))

    x = channel_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), padding='same',
                     activation='relu', name=name + '_1')
    x = channel_unit(x, 128, (5, 5), padding='same', activation='relu', name=name + '_2')

    x = channel_unit(x, 192, (3, 3), padding='same', activation='relu', name=name + '_3')
    x = channel_unit(x, 192, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_4')
    x = channel_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_5')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(4096, activation='relu', name=name + '_6_dense')(fc)
        fc = Dense(4096, activation='relu', name=name + '_7_dense')(fc)
        fc = Dense(1000, activation='softmax', name=name + '_8_dense')(fc)
        x = fc

    return Model(img, x)


def original(include_fc=True):
    name = NAME + '_original'
    img = Input(shape=(220, 220, 3))

    x = conv_unit(img, 48, (11, 11), max_pooling=False, strides=(4, 4), activation='relu', name=name + '_1')
    x = conv_unit(x, 128, (5, 5), activation='relu', name=name + '_2')

    x = conv_unit(x, 192, (3, 3), activation='relu', name=name + '_3')
    x = conv_unit(x, 192, (3, 3), max_pooling=False, activation='relu', name=name + '_4')
    x = conv_unit(x, 128, (3, 3), max_pooling=False, activation='relu', name=name + '_5')

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    x = Flatten()(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(4096, activation='relu', name=name + '_6_dense')(fc)
        fc = Dense(4096, activation='relu', name=name + '_7_dense')(fc)
        fc = Dense(1000, activation='softmax', name=name + '_8_dense')(fc)
        x = fc

    return Model(img, x)


def fc1():
    """ First separated fully connected layer. """
    block_input = Input(shape=(6272,))
    layer = Dense(2048, activation='relu')(block_input)
    return Model(block_input, layer)


def fc2():
    """ Second fully connected layer. """
    block_input = Input(shape=(4096,))
    layer = Dense(4096, activation='relu')(block_input)
    layer = Dense(1000, activation='softmax')(layer)
    return Model(block_input, layer)
