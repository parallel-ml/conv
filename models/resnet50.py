from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, \
    ZeroPadding2D, Activation, AveragePooling2D
from keras.models import Model
from keras import layers
from unit import filter_unit, channel_unit, xy_unit


MODE = 'original'


def original():
    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity(x, 3, [64, 64, 256])
    x = identity(x, 3, [64, 64, 256])

    x = conv(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])

    x = conv(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])

    x = conv(x, 3, [512, 512, 2048])
    x = identity(x, 3, [512, 512, 2048])
    x = identity(x, 3, [512, 512, 2048])

    x = AveragePooling2D((7, 7))(x)

    fc = Flatten()(x)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img_input, fc)


def filter():
    global MODE
    MODE = 'filter'

    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = filter_unit(x, 64, (7, 7), strides=(2, 2), activation='relu')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity(x, 3, [64, 64, 256])
    x = identity(x, 3, [64, 64, 256])

    x = conv(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])

    x = conv(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])

    x = conv(x, 3, [512, 512, 2048])
    x = identity(x, 3, [512, 512, 2048])
    x = identity(x, 3, [512, 512, 2048])

    x = AveragePooling2D((7, 7))(x)

    fc = Flatten()(x)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img_input, fc)


def xy():
    global MODE
    MODE = 'xy'

    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = xy_unit(x, 64, (7, 7), strides=(2, 2), activation='relu')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity(x, 3, [64, 64, 256])
    x = identity(x, 3, [64, 64, 256])

    x = conv(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])
    x = identity(x, 3, [128, 128, 512])

    x = conv(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])
    x = identity(x, 3, [256, 256, 1024])

    x = conv(x, 3, [512, 512, 2048])
    x = identity(x, 3, [512, 512, 2048])
    x = identity(x, 3, [512, 512, 2048])

    x = AveragePooling2D((7, 7))(x)

    fc = Flatten()(x)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img_input, fc)


def channel():
    global MODE
    MODE = 'channel'

    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = channel_unit(x, 64, (7, 7), strides=(2, 2), max_pooling=False, padding='valid', activation='relu')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, (3, 3), [64, 64, 256], strides=(1, 1))
    x = identity(x, (3, 3), [64, 64, 256])
    x = identity(x, (3, 3), [64, 64, 256])

    x = conv(x, (3, 3), [128, 128, 512])
    x = identity(x, (3, 3), [128, 128, 512])
    x = identity(x, (3, 3), [128, 128, 512])
    x = identity(x, (3, 3), [128, 128, 512])

    x = conv(x, (3, 3), [256, 256, 1024])
    x = identity(x, (3, 3), [256, 256, 1024])
    x = identity(x, (3, 3), [256, 256, 1024])
    x = identity(x, (3, 3), [256, 256, 1024])
    x = identity(x, (3, 3), [256, 256, 1024])
    x = identity(x, (3, 3), [256, 256, 1024])

    x = conv(x, (3, 3), [512, 512, 2048])
    x = identity(x, (3, 3), [512, 512, 2048])
    x = identity(x, (3, 3), [512, 512, 2048])

    x = AveragePooling2D((7, 7))(x)

    fc = Flatten()(x)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img_input, fc)


def conv(input_tensor, kernel_size, filters, strides=(2, 2)):
    f1, f2, f3 = filters

    if MODE == 'channel':
        x = channel_unit(input_tensor, f1, (1, 1), max_pooling=False, strides=strides, padding='valid')
        x = channel_unit(x, f2, kernel_size, max_pooling=False, padding='same')
        x = channel_unit(x, f3, (1, 1), max_pooling=False, padding='valid', activation=None)

        shortcut = channel_unit(input_tensor, f3, (1, 1), max_pooling=False, strides=strides,
                                padding='valid', activation=None)
    elif MODE == 'filter':
        x = filter_unit(input_tensor, f1, (1, 1), max_pooling=False, strides=strides, padding='valid')
        x = filter_unit(x, f2, kernel_size, max_pooling=False, padding='same')
        x = filter_unit(x, f3, (1, 1), max_pooling=False, padding='valid', activation=None)

        shortcut = filter_unit(input_tensor, f3, (1, 1), max_pooling=False, strides=strides,
                               padding='valid', activation=None)
    elif MODE == 'xy':
        x = xy_unit(input_tensor, f1, (1, 1), max_pooling=False, strides=strides, padding='valid')
        x = xy_unit(x, f2, kernel_size, max_pooling=False, padding='same')
        x = xy_unit(x, f3, (1, 1), max_pooling=False, padding='valid', activation=None)

        shortcut = xy_unit(input_tensor, f3, (1, 1), max_pooling=False, strides=strides,
                           padding='valid', activation=None)
    else:
        x = Conv2D(f1, (1, 1), strides=strides, activation='relu')(input_tensor)
        x = Conv2D(f2, kernel_size, padding='same', activation='relu')(x)
        x = Conv2D(f3, (1, 1))(x)

        shortcut = Conv2D(f3, (1, 1), strides=strides)(input_tensor)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity(input_tensor, kernel_size, filters):
    f1, f2, f3 = filters

    if MODE == 'channel':
        x = channel_unit(input_tensor, f1, (1, 1), max_pooling=False, padding='valid')
        x = channel_unit(x, f2, kernel_size, max_pooling=False, padding='same')
        x = channel_unit(x, f3, (1, 1), max_pooling=False, padding='valid', activation=None)
    elif MODE == 'filter':
        x = filter_unit(input_tensor, f1, (1, 1), max_pooling=False, padding='valid')
        x = filter_unit(x, f2, kernel_size, max_pooling=False, padding='same')
        x = filter_unit(x, f3, (1, 1), max_pooling=False, padding='valid', activation=None)
    elif MODE == 'xy':
        x = xy_unit(input_tensor, f1, (1, 1), max_pooling=False, padding='valid')
        x = xy_unit(x, f2, kernel_size, max_pooling=False, padding='same')
        x = xy_unit(x, f3, (1, 1), max_pooling=False, padding='valid', activation=None)
    else:
        x = Conv2D(f1, (1, 1), activation='relu')(input_tensor)
        x = Conv2D(f2, kernel_size, padding='same', activation='relu')(x)
        x = Conv2D(f3, (1, 1))(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
