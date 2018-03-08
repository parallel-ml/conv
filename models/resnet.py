from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, \
    ZeroPadding2D, Activation, AveragePooling2D
from keras.models import Model
from keras import layers
from unit import filter_unit, channel_unit, xy_unit


def original():
    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))
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


def conv(input_tensor, kernel_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), strides=strides, activation='relu')(input_tensor)
    x = Conv2D(filters2, kernel_size, padding='same', activation='relu')(x)
    x = Conv2D(filters3, (1, 1), activation='relu')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, activation='relu')(input_tensor)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), activation='relu')(input_tensor)
    x = Conv2D(filters2, kernel_size, padding='same', activation='relu')(x)
    x = Conv2D(filters3, (1, 1))(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
