from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import Multiply, Concatenate
from keras.models import Model
from keras.layers.core import Layer
from keras import backend as K


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


def conv2D_bn(x, nb_filter, nb_row, nb_col, activation='relu', batch_norm=True):
    x = Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation,)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    if batch_norm:
        x = LRN2D()(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
    return x


def alexnet():
    img_input = Input(shape=(224, 224, 3))

    stream1 = conv2D_bn(img_input, 3, 11, 11)
    stream1 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream1)
    stream1 = ZeroPadding2D(padding=(1, 1))(stream1)

    stream1 = conv2D_bn(stream1, 48, 5, 5)
    stream1 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream1)
    stream1 = ZeroPadding2D(padding=(1, 1))(stream1)

    stream1 = conv2D_bn(stream1, 128, 3, 3)
    stream1 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream1)
    stream1 = ZeroPadding2D(padding=(1, 1))(stream1)

    stream2 = conv2D_bn(img_input, 3, 11, 11)
    stream2 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream2)
    stream2 = ZeroPadding2D(padding=(1, 1))(stream2)

    stream2 = conv2D_bn(stream2, 48, 5, 5)
    stream2 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream2)
    stream2 = ZeroPadding2D(padding=(1, 1))(stream2)

    stream2 = conv2D_bn(stream2, 128, 3, 3)
    stream2 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream2)
    stream2 = ZeroPadding2D(padding=(1, 1))(stream2)

    stream1_after_merge = Concatenate(axis=3)([stream1, stream2])
    stream1_after_merge = ZeroPadding2D(padding=(1, 1))(stream1_after_merge)

    stream2_after_merge = Concatenate(axis=3)([stream1, stream2])
    stream2_after_merge = ZeroPadding2D(padding=(1, 1))(stream2_after_merge)

    stream1_after_merge = conv2D_bn(stream1_after_merge, 192, 3, 3)
    stream1_after_merge = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream1_after_merge)
    stream1_after_merge = ZeroPadding2D(padding=(1, 1))(stream1_after_merge)

    stream1_after_merge = conv2D_bn(stream1_after_merge, 192, 3, 3)
    stream1_after_merge = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream1_after_merge)
    stream1_after_merge = ZeroPadding2D(padding=(1, 1))(stream1_after_merge)

    stream2_after_merge = conv2D_bn(stream2_after_merge, 192, 3, 3)
    stream2_after_merge = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream2_after_merge)
    stream2_after_merge = ZeroPadding2D(padding=(1, 1))(stream2_after_merge)

    stream2_after_merge = conv2D_bn(stream2_after_merge, 192, 3, 3)
    stream2_after_merge = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(stream2_after_merge)
    stream2_after_merge = ZeroPadding2D(padding=(1, 1))(stream2_after_merge)

    fc1_stream1 = Multiply()([stream1_after_merge, stream2_after_merge])
    fc1_stream1 = Flatten()(fc1_stream1)
    fc1_stream1 = Dense(2048, activation='relu')(fc1_stream1)
    fc1_stream1 = Dropout(0.5)(fc1_stream1)

    fc1_stream2 = Multiply()([stream1_after_merge, stream2_after_merge])
    fc1_stream2 = Flatten()(fc1_stream2)
    fc1_stream2 = Dense(2048, activation='relu')(fc1_stream2)
    fc1_stream2 = Dropout(0.5)(fc1_stream2)

    fc2_stream1 = Multiply()([fc1_stream1, fc1_stream2])
    fc2_stream1 = Dense(2048, activation='relu')(fc2_stream1)
    fc2_stream1 = Dropout(0.5)(fc2_stream1)

    fc2_stream2 = Multiply()([fc1_stream1, fc1_stream2])
    fc2_stream2 = Dense(2048, activation='relu')(fc2_stream2)
    fc2_stream2 = Dropout(0.5)(fc2_stream2)

    fc = Multiply()([fc2_stream1, fc2_stream2])
    fc = Dense(output_dim=1000, activation='softmax')(fc)

    return Model(input=img_input, output=[fc])


model = alexnet().summary()
