from keras.layers import Conv2D, MaxPooling2D, Dense, Input, SeparableConv2D, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras import layers
from utils import separable


NAME = 'xception'


def original():
    img_input = Input([224, 224, 3])

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, activation='relu')(img_input)
    x = Conv2D(64, (3, 3), use_bias=False, activation='relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    return Model(img_input, x)


def filter():
    img_input = Input([224, 224, 3])
    name = NAME + '_filter'

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, activation='relu')(img_input)
    x = Conv2D(64, (3, 3), use_bias=False, activation='relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False, name=name + '_1_conv')(x)
    x = separable.filter(x, 128, (3, 3), padding='same', use_bias=False, name=name + '_2')
    x = Activation('relu')(x)
    x = separable.filter(x, 128, (3, 3), padding='same', use_bias=False, name=name + '_3')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name=name + '_4_conv')(x)
    x = Activation('relu')(x)
    x = separable.filter(x, 256, (3, 3), padding='same', use_bias=False, name=name + '_5')
    x = Activation('relu')(x)
    x = separable.filter(x, 256, (3, 3), padding='same', use_bias=False, name=name + '_6')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False, name=name + '_7_conv')(x)
    x = Activation('relu')(x)
    x = separable.filter(x, 728, (3, 3), padding='same', use_bias=False, name=name + '_8')
    x = Activation('relu')(x)
    x = separable.filter(x, 728, (3, 3), padding='same', use_bias=False, name=name + '_9')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    for i in range(8):
        depth = i * 3 + 10
        residual = x
        x = Activation('relu')(x)
        x = separable.filter(x, 728, (3, 3), padding='same', name=name + '_' + str(depth))
        x = Activation('relu')(x)
        x = separable.filter(x, 728, (3, 3), padding='same', name=name + '_' + str(depth))
        x = Activation('relu')(x)
        x = separable.filter(x, 728, (3, 3), padding='same', name=name + '_' + str(depth))
        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False, name=name + '_35_conv')(x)
    x = Activation('relu')(x)
    x = separable.filter(x, 728, (3, 3), padding='same', use_bias=False, name=name + '_36')
    x = Activation('relu')(x)
    x = separable.filter(x, 1024, (3, 3), padding='same', use_bias=False, name=name + '_37')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = separable.filter(x, 1536, (3, 3), padding='same', use_bias=False, name=name + '_38')
    x = Activation('relu')(x)
    x = separable.filter(x, 2048, (3, 3), padding='same', use_bias=False, name=name + '_39')
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax', name=name + '_40_dense')(x)

    return Model(img_input, x)
