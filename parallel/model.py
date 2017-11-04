from keras.layers import Input
from keras.layers.convolutional import Conv2D
# from keras.activations import Dense, Activation
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.models import Model
from keras.models import Sequential


# input_dim_ordering should be set to "tf"
def load_spatial():
    return load_cnn(nb_channel=3)


def load_temporal():
    return load_cnn(nb_channel=6)


def load_cnn(nb_class=1000, bias=True, act='relu', bn=True, dropout=False, moredense=False, nb_filter=256,
             nb_channel=3, clsfy_act='softmax'):
    # input image is 16x12, RGB(x3)
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), use_bias=bias, padding='same', input_shape=(12, 16, nb_channel)))
    # now model.output_shape == (None, 12, 16, 128)
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_1'))

    model.add(Conv2D(nb_filter, (3, 3), use_bias=bias, padding='same'))
    # now model.output_shape == (None, 12, 16, 128)
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_2'))

    model.add(Conv2D(nb_filter, (3, 3), use_bias=bias, padding='same'))
    # now model.output_shape == (None, 12, 16, 128)
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_3'))

    model.add(Flatten())
    # now model.output_shape == (None, 24576)
    model.add(Dense(256, input_dim=24576, use_bias=bias, name='dense_4'))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_4'))

    return model


def load_fc(split=1):
    input_shape = 7680 / split
    layer_shape = 4096 / split
    model = Sequential()
    model.add(Dense(layer_shape, input_shape=(input_shape,)))
    model.add(BatchNormalization(input_shape=(layer_shape,)))
    model.add(Activation('relu', input_shape=(layer_shape,)))

    model.add(Dense(layer_shape, input_shape=(layer_shape,)))
    model.add(BatchNormalization(input_shape=(layer_shape,)))
    model.add(Activation('relu', input_shape=(layer_shape,)))

    model.add(Dense(51, input_shape=(layer_shape,)))
    model.add(BatchNormalization(input_shape=(51,)))
    model.add(Activation('softmax', input_shape=(51,)))
    return model


def load_maxpool(input_shape=(16, 256), N=16):
    input = Input(shape=input_shape, name='input')

    max1 = MaxPooling1D(pool_size=N, strides=N)(input)
    max2 = MaxPooling1D(pool_size=N / 2, strides=N / 2)(input)
    max3 = MaxPooling1D(pool_size=N / 4, strides=N / 4)(input)
    max4 = MaxPooling1D(pool_size=N / 8, strides=N / 8)(input)

    mrg = Concatenate(axis=1)([max1, max2, max3, max4])

    flat = Flatten()(mrg)

    model = Model(input=input, outputs=flat)

    return model
