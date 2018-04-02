from keras.layers import Input, Conv2D, MaxPooling2D, Lambda
from keras.models import Model
from keras.layers.merge import concatenate
import tensorflow as tf

NAME = 'yolo'


def original():
    name = NAME + '_original'
    img_input = Input([224, 224, 3])
    x = Conv2D(32, (3, 3), padding='same', name=name + '_1_conv', use_bias=False)(img_input)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), padding='same', name=name + '_2_conv', use_bias=False)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', name=name + '_3_conv', use_bias=False)(x)
    x = Conv2D(64, (1, 1), padding='same', name=name + '_4_conv', use_bias=False)(x)
    x = Conv2D(128, (3, 3), padding='same', name=name + '_5_conv', use_bias=False)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3, 3), padding='same', name=name + '_6_conv', use_bias=False)(x)
    x = Conv2D(128, (1, 1), padding='same', name=name + '_7_conv', use_bias=False)(x)
    x = Conv2D(256, (3, 3), padding='same', name=name + '_8_conv', use_bias=False)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', name=name + '_9_conv', use_bias=False)(x)
    x = Conv2D(256, (1, 1), padding='same', name=name + '_10_conv', use_bias=False)(x)
    x = Conv2D(512, (3, 3), padding='same', name=name + '_11_conv', use_bias=False)(x)
    x = Conv2D(256, (1, 1), padding='same', name=name + '_12_conv', use_bias=False)(x)
    x = Conv2D(512, (3, 3), padding='same', name=name + '_13_conv', use_bias=False)(x)

    skip_connection = x

    x = MaxPooling2D()(x)

    x = Conv2D(1024, (3, 3), padding='same', name=name + '_14_conv', use_bias=False)(x)
    x = Conv2D(512, (1, 1), padding='same', name=name + '_15_conv', use_bias=False)(x)
    x = Conv2D(1024, (3, 3), padding='same', name=name + '_16_conv', use_bias=False)(x)
    x = Conv2D(512, (1, 1), padding='same', name=name + '_17_conv', use_bias=False)(x)
    x = Conv2D(1024, (3, 3), padding='same', name=name + '_18_conv', use_bias=False)(x)
    x = Conv2D(1024, (3, 3), padding='same', name=name + '_19_conv', use_bias=False)(x)
    x = Conv2D(1024, (3, 3), padding='same', name=name + '_20_conv', use_bias=False)(x)

    skip_connection = Conv2D(64, (1, 1), padding='same', name=name + '_21_conv', use_bias=False)(
        skip_connection)
    skip_connection = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(skip_connection)
    x = concatenate([skip_connection, x])

    x = Conv2D(1024, (3, 3), padding='same', name=name + '_22_conv', use_bias=False)(x)

    # use 1000 for imagenet classification
    x = Conv2D((4 + 1 + 1000) * 5, (1, 1), padding='same', name=name + '_23_conv')(x)

    return Model(img_input, x)
