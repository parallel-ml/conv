from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, \
    ZeroPadding2D, Activation, AveragePooling2D
from keras.models import Model
from keras import layers
from unit import filter_unit, channel_unit, xy_unit

MODE = 'original'
NAME = 'resnet50'


def original(include_fc=True):
    name = NAME + '_original'
    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', name=name + '_1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, 3, [64, 64, 256], strides=(1, 1), name=name + '_2')
    x = identity(x, 3, [64, 64, 256], name=name + '_3')
    x = identity(x, 3, [64, 64, 256], name=name + '_4')

    x = conv(x, 3, [128, 128, 512], name=name + '_5')
    x = identity(x, 3, [128, 128, 512], name=name + '_6')
    x = identity(x, 3, [128, 128, 512], name=name + '_7')
    x = identity(x, 3, [128, 128, 512], name=name + '_8')

    x = conv(x, 3, [256, 256, 1024], name=name + '_9')
    x = identity(x, 3, [256, 256, 1024], name=name + '_10')
    x = identity(x, 3, [256, 256, 1024], name=name + '_11')
    x = identity(x, 3, [256, 256, 1024], name=name + '_12')
    x = identity(x, 3, [256, 256, 1024], name=name + '_13')
    x = identity(x, 3, [256, 256, 1024], name=name + '_14')

    x = conv(x, 3, [512, 512, 2048], name=name + '_15')
    x = identity(x, 3, [512, 512, 2048], name=name + '_16')
    x = identity(x, 3, [512, 512, 2048], name=name + '_17')

    x = AveragePooling2D((7, 7))(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(1000, activation='softmax', name=name + '_18_dense')(fc)
        x = fc

    return Model(img_input, x)


def filter(include_fc=True):
    global MODE
    MODE = 'filter'

    name = NAME + '_filter'
    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', name=name + '_1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, 3, [64, 64, 256], strides=(1, 1), name=name + '_2')
    x = identity(x, 3, [64, 64, 256], name=name + '_3')
    x = identity(x, 3, [64, 64, 256], name=name + '_4')

    x = conv(x, 3, [128, 128, 512], name=name + '_5')
    x = identity(x, 3, [128, 128, 512], name=name + '_6')
    x = identity(x, 3, [128, 128, 512], name=name + '_7')
    x = identity(x, 3, [128, 128, 512], name=name + '_8')

    x = conv(x, 3, [256, 256, 1024], name=name + '_9')
    x = identity(x, 3, [256, 256, 1024], name=name + '_10')
    x = identity(x, 3, [256, 256, 1024], name=name + '_11')
    x = identity(x, 3, [256, 256, 1024], name=name + '_12')
    x = identity(x, 3, [256, 256, 1024], name=name + '_13')
    x = identity(x, 3, [256, 256, 1024], name=name + '_14')

    x = conv(x, 3, [512, 512, 2048], name=name + '_15')
    x = identity(x, 3, [512, 512, 2048], name=name + '_16')
    x = identity(x, 3, [512, 512, 2048], name=name + '_17')

    x = AveragePooling2D((7, 7))(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(1000, activation='softmax', name=name + '_18_dense')(fc)
        x = fc

    return Model(img_input, x)


def xy(include_fc=True):
    global MODE
    MODE = 'xy'

    name = NAME + '_spatial'
    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', name=name + '_1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, 3, [64, 64, 256], strides=(1, 1), name=name + '_2')
    x = identity(x, 3, [64, 64, 256], name=name + '_3')
    x = identity(x, 3, [64, 64, 256], name=name + '_4')

    x = conv(x, 3, [128, 128, 512], name=name + '_5')
    x = identity(x, 3, [128, 128, 512], name=name + '_6')
    x = identity(x, 3, [128, 128, 512], name=name + '_7')
    x = identity(x, 3, [128, 128, 512], name=name + '_8')

    x = conv(x, 3, [256, 256, 1024], name=name + '_9')
    x = identity(x, 3, [256, 256, 1024], name=name + '_10')
    x = identity(x, 3, [256, 256, 1024], name=name + '_11')
    x = identity(x, 3, [256, 256, 1024], name=name + '_12')
    x = identity(x, 3, [256, 256, 1024], name=name + '_13')
    x = identity(x, 3, [256, 256, 1024], name=name + '_14')

    x = conv(x, 3, [512, 512, 2048], name=name + '_15')
    x = identity(x, 3, [512, 512, 2048], name=name + '_16')
    x = identity(x, 3, [512, 512, 2048], name=name + '_17')

    x = AveragePooling2D((7, 7))(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(1000, activation='softmax', name=name + '_18_dense')(fc)
        x = fc

    return Model(img_input, x)


def channel(include_fc=True):
    global MODE
    MODE = 'channel'

    name = NAME + '_channel'
    img_input = Input([224, 224, 3])

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', name=name + '_1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv(x, 3, [64, 64, 256], strides=(1, 1), name=name + '_2')
    x = identity(x, 3, [64, 64, 256], name=name + '_3')
    x = identity(x, 3, [64, 64, 256], name=name + '_4')

    x = conv(x, 3, [128, 128, 512], name=name + '_5')
    x = identity(x, 3, [128, 128, 512], name=name + '_6')
    x = identity(x, 3, [128, 128, 512], name=name + '_7')
    x = identity(x, 3, [128, 128, 512], name=name + '_8')

    x = conv(x, 3, [256, 256, 1024], name=name + '_9')
    x = identity(x, 3, [256, 256, 1024], name=name + '_10')
    x = identity(x, 3, [256, 256, 1024], name=name + '_11')
    x = identity(x, 3, [256, 256, 1024], name=name + '_12')
    x = identity(x, 3, [256, 256, 1024], name=name + '_13')
    x = identity(x, 3, [256, 256, 1024], name=name + '_14')

    x = conv(x, 3, [512, 512, 2048], name=name + '_15')
    x = identity(x, 3, [512, 512, 2048], name=name + '_16')
    x = identity(x, 3, [512, 512, 2048], name=name + '_17')

    x = AveragePooling2D((7, 7))(x)

    if include_fc:
        fc = Flatten()(x)
        fc = Dense(1000, activation='softmax', name=name + '_18_dense')(fc)
        x = fc

    return Model(img_input, x)


def conv(input_tensor, kernel_size, filters, name, strides=(2, 2)):
    f1, f2, f3 = filters

    if MODE == 'channel':
        x = channel_unit(input_tensor, f1, (1, 1), max_pooling=False, strides=strides, padding='valid',
                         activation='relu', name=name + 'a')
        x = channel_unit(x, f2, (kernel_size, kernel_size), max_pooling=False, padding='same',
                         activation='relu', name=name + 'v')
        x = channel_unit(x, f3, (1, 1), max_pooling=False, padding='valid', name=name + 'c')
        shortcut = channel_unit(input_tensor, f3, (1, 1), max_pooling=False, strides=strides,
                                padding='valid', name=name + 's')

    elif MODE == 'filter':
        x = filter_unit(input_tensor, f1, (1, 1), max_pooling=False, strides=strides, padding='valid',
                        activation='relu', name=name + 'a')
        x = filter_unit(x, f2, (kernel_size, kernel_size), max_pooling=False, padding='same',
                        activation='relu', name=name + 'b')
        x = filter_unit(x, f3, (1, 1), max_pooling=False, padding='valid', name=name + 'c')
        shortcut = filter_unit(input_tensor, f3, (1, 1), max_pooling=False, strides=strides,
                               padding='valid', name=name + 's')

    elif MODE == 'xy':
        x = xy_unit(input_tensor, f1, (1, 1), max_pooling=False, strides=strides, padding='valid',
                    activation='relu', name=name + 'a')
        x = xy_unit(x, f2, (kernel_size, kernel_size), max_pooling=False, padding='same',
                    activation='relu', name=name + 'b')
        x = xy_unit(x, f3, (1, 1), max_pooling=False, padding='valid', name=name + 'c')
        shortcut = xy_unit(input_tensor, f3, (1, 1), max_pooling=False, strides=strides,
                           padding='valid', name=name + 's')

    else:
        x = Conv2D(f1, (1, 1), strides=strides, activation='relu', name=name + 'a_conv')(input_tensor)
        x = Conv2D(f2, (kernel_size, kernel_size), padding='same', activation='relu', name=name + 'b_conv')(x)
        x = Conv2D(f3, (1, 1), name=name + 'c_conv')(x)
        shortcut = Conv2D(f3, (1, 1), strides=strides, name=name + 's_conv')(input_tensor)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity(input_tensor, kernel_size, filters, name):
    f1, f2, f3 = filters

    if MODE == 'channel':
        x = channel_unit(input_tensor, f1, (1, 1), max_pooling=False, padding='valid',
                         activation='relu', name=name + 'a')
        x = channel_unit(x, f2, (kernel_size, kernel_size), max_pooling=False, padding='same', name=name + 'b')
        x = channel_unit(x, f3, (1, 1), max_pooling=False, padding='valid', name=name + 'c')
    elif MODE == 'filter':
        x = filter_unit(input_tensor, f1, (1, 1), max_pooling=False, padding='valid',
                        activation='relu', name=name + 'a')
        x = filter_unit(x, f2, (kernel_size, kernel_size), max_pooling=False, padding='same', name=name + 'b')
        x = filter_unit(x, f3, (1, 1), max_pooling=False, padding='valid', name=name + 'c')
    elif MODE == 'xy':
        x = xy_unit(input_tensor, f1, (1, 1), max_pooling=False, padding='valid',
                    activation='relu', name=name + 'a')
        x = xy_unit(x, f2, (kernel_size, kernel_size), max_pooling=False, padding='same', name=name + 'b')
        x = xy_unit(x, f3, (1, 1), max_pooling=False, padding='valid', name=name + 'c')
    else:
        x = Conv2D(f1, (1, 1), activation='relu', name=name + 'a_conv')(input_tensor)
        x = Conv2D(f2, (kernel_size, kernel_size), padding='same', activation='relu', name=name + 'b_conv')(x)
        x = Conv2D(f3, (1, 1), name=name + 'c_conv')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
