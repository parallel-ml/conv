import numpy as np
import time
import os
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import Multiply, Concatenate
from keras.models import Model
from keras.layers.core import Layer
from keras import backend as K


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img_input = None


class LRN2D(Layer):
    """ LRN2D class is from original keras but gets removed at latest version """
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN2D, self).__init__(**kwargs)

    def get_output(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2
        input_sqr = K.square(x)
        extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
        input_sqr = K.concatenate(
            [extra_channels[:, :half_n, :, :], input_sqr, extra_channels[:, half_n + int(ch):, :, :]], axis=1)

        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + int(ch), :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv2D_bn(x, nb_filter, nb_row, nb_col, activation='relu', batch_norm=True, name=''):
    if name != '':
        global img_input
        if img_input != x:
            input_shape = Model(img_input, x).output_shape
            input_shape = input_shape[1:] if input_shape[0] is None else input_shape
        else:
            input_shape = (224, 224, 3)
        temp_input = Input(shape=input_shape)

        y = Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, )(temp_input)
        y = ZeroPadding2D(padding=(1, 1))(y)
        if batch_norm:
            y = LRN2D()(y)
            y = ZeroPadding2D(padding=(1, 1))(y)
        y = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(y)
        y = ZeroPadding2D(padding=(1, 1))(y)

        temp_model = Model(temp_input, y)
        test_x = np.random.random_sample(input_shape)
        start = time.time()
        for _ in range(50):
            temp_model.predict(np.array([test_x]))
        print '{:s}: {:.3f} sec'.format(name, (time.time() - start) / 50)
        del temp_model

    x = Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation,)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    if batch_norm:
        x = LRN2D()(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x


def concatenate(s1, s2, name=''):
    if name != '':
        global img_input
        input_shape = Model(img_input, s1).output_shape
        input_shape = input_shape[1:] if input_shape[0] is None else input_shape
        temp_input1 = Input(shape=input_shape)
        temp_input2 = Input(shape=input_shape)

        y = Concatenate(axis=3)([temp_input1, temp_input2])
        y = ZeroPadding2D(padding=(1, 1))(y)

        temp_model = Model([temp_input1, temp_input2], y)
        test_s1 = np.random.random_sample(input_shape)
        test_s2 = np.random.random_sample(input_shape)
        start = time.time()
        for _ in range(50):
            temp_model.predict([np.array([test_s1]), np.array([test_s2])])
        print '{:s}: {:.3f} sec'.format(name, (time.time() - start) / 50)
        del temp_model

    x = Concatenate(axis=3)([s1, s2])
    x = ZeroPadding2D(padding=(1, 1))(x)
    return x


def multiply(s1, s2, flatten=True, name=''):
    if name != '':
        global img_input
        input_shape = Model(img_input, s1).output_shape
        input_shape = input_shape[1:] if input_shape[0] is None else input_shape
        temp_input1 = Input(shape=input_shape)
        temp_input2 = Input(shape=input_shape)

        y = Multiply()([temp_input1, temp_input2])
        if flatten:
            y = Flatten()(y)

        temp_model = Model([temp_input1, temp_input2], y)
        test_s1 = np.random.random_sample(input_shape)
        test_s2 = np.random.random_sample(input_shape)
        start = time.time()
        for _ in range(50):
            temp_model.predict([np.array([test_s1]), np.array([test_s2])])
        print '{:s}: {:.3f} sec'.format(name, (time.time() - start) / 50)
        del temp_model

    x = Multiply()([s1, s2])
    if flatten:
        x = Flatten()(x)
    return x


def dense(x, act='relu', dim=2048, dropout=True, name=''):
    if name != '':
        global img_input
        input_shape = Model(img_input, x).output_shape
        input_shape = input_shape[1:] if input_shape[0] is None else input_shape
        temp_input = Input(shape=input_shape)

        y = Dense(dim, activation=act)(temp_input)
        if dropout:
            y = Dropout(0.5)(y)

        temp_model = Model(temp_input, y)
        test_x = np.random.random_sample(input_shape)
        start = time.time()
        for _ in range(50):
            temp_model.predict(np.array([test_x]))
        print '{:s}: {:.3f} sec'.format(name, (time.time() - start) / 50)
        del temp_model

    x = Dense(dim, activation=act)(x)
    if dropout:
        x = Dropout(0.5)(x)
    return x


def alexnet():
    global img_input
    img_input = Input(shape=(224, 224, 3))

    # first three conv nets
    stream1 = conv2D_bn(img_input, 3, 11, 11, name='conv1 single stream')
    stream1 = conv2D_bn(stream1, 48, 5, 5, name='conv2 single stream')
    stream1 = conv2D_bn(stream1, 128, 3, 3, name='conv3 single stream')

    stream2 = conv2D_bn(img_input, 3, 11, 11)
    stream2 = conv2D_bn(stream2, 48, 5, 5)
    stream2 = conv2D_bn(stream2, 128, 3, 3)

    # merge conv nets
    stream1_after_merge = concatenate(stream1, stream2, name='concatenate')
    stream2_after_merge = concatenate(stream1, stream2)

    # rest two conv nets
    stream1_after_merge = conv2D_bn(stream1_after_merge, 192, 3, 3, name='conv4 single stream')
    stream1_after_merge = conv2D_bn(stream1_after_merge, 192, 3, 3, name='conv5 single stream')

    stream2_after_merge = conv2D_bn(stream2_after_merge, 192, 3, 3)
    stream2_after_merge = conv2D_bn(stream2_after_merge, 192, 3, 3)

    # first fc layer
    fc1_stream1 = multiply(stream1_after_merge, stream2_after_merge, flatten=True, name='merge1_fc')
    fc1_stream1 = dense(fc1_stream1, name='fc1 single stream')

    fc1_stream2 = multiply(stream1_after_merge, stream2_after_merge, flatten=True)
    fc1_stream2 = dense(fc1_stream2)

    # second fc layer
    fc2_stream1 = multiply(fc1_stream1, fc1_stream2, flatten=False, name='merge2_fc')
    fc2_stream1 = dense(fc2_stream1, name='fc2 single stream')

    fc2_stream2 = multiply(fc1_stream1, fc1_stream2, flatten=False)
    fc2_stream2 = dense(fc2_stream2)

    # final classification layer
    fc = multiply(fc2_stream1, fc2_stream2, flatten=False, name='merge3_fc')
    fc = dense(fc, act='softmax', dim=1000, dropout=False, name='fc3 single stream')

    return Model(img_input, fc)


model = alexnet()
