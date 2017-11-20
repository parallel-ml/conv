from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Input, Flatten, Dense
from keras.models import Model
from keras.layers.core import Layer
import keras.backend as K


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


def node_6_block1():
    img = Input(shape=(224, 224, 3))
    output = conv2D_bn(img, 3, 11, 11)
    model = Model(img, output)
    return model


def node_6_block2():
    block_input = Input(shape=(111, 111, 3))
    x = conv2D_bn(block_input, 48, 5, 5)
    x = conv2D_bn(x, 128, 3, 3)
    x = conv2D_bn(x, 192, 3, 3)
    x = conv2D_bn(x, 192, 3, 3)
    x = Flatten()(x)
    model = Model(block_input, x)
    return model


def node_6_block3():
    block_input = Input(shape=(27648,))
    x = Dense(2048, activation='relu')(block_input)
    model = Model(block_input, x)
    return model


def node_6_block4():
    block_input = Input(shape=(4096,))
    x = Dense(4096, activation='relu')(block_input)
    x = Dense(1000, activation='relu')(block_input)
    model = Model(block_input, x)
    return model


def node_8_block2():
    block_input = Input(shape=(111, 111, 3))
    x = conv2D_bn(block_input, 48, 5, 5)
    model = Model(block_input, x)
    return model


def node_8_block3():
    block_input = Input(shape=(57, 57, 48))
    x = conv2D_bn(block_input, 128, 3, 3)
    x = conv2D_bn(x, 192, 3, 3)
    x = conv2D_bn(x, 192, 3, 3)
    x = Flatten()(x)
    model = Model(block_input, x)
    return model


def conv2D_bn(x, nb_filter, nb_row, nb_col, activation='relu', batch_norm=True):
    x = Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, )(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    if batch_norm:
        x = LRN2D()(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    return x


def whole():
    img_input = Input(shape=(224, 224, 3))

    stream1 = conv2D_bn(img_input, 3, 11, 11)
    stream1 = conv2D_bn(stream1, 48, 5, 5)
    stream1 = conv2D_bn(stream1, 128, 3, 3)

    stream1 = conv2D_bn(stream1, 192, 3, 3)
    stream1 = conv2D_bn(stream1, 192, 3, 3)

    fc = Flatten()(stream1)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img_input, fc)