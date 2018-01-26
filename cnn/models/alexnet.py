from keras import backend as K
from keras.layers import Flatten, Dense, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Layer
from keras.models import Model


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


def conv2D_bn(x, nb_filter, nb_row, nb_col, activation='relu', batch_norm=True, max_pooling=True, stride=(1, 1)):
    if max_pooling:
        x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    x = Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, strides=stride, padding='same')(x)

    if batch_norm:
        x = LRN2D()(x)
    return x


def xy_unit():
    pass


def channel_unit():
    pass


def filter_unit():
    pass


def alexnet():
    img = Input(shape=(220, 220, 3))

    x = conv2D_bn(img, 48, 11, 11, max_pooling=False, stride=(4, 4))
    x = conv2D_bn(x, 128, 5, 5)

    x = conv2D_bn(x, 192, 3, 3)
    x = conv2D_bn(x, 192, 3, 3, max_pooling=False)
    x = conv2D_bn(x, 128, 3, 3, max_pooling=False)

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(1000, activation='softmax')(fc)

    return Model(img, fc)
