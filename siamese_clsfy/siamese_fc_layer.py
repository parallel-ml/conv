import os

import numpy as np
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from util.output import title, timer, avg_timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    run_fc_first()
    run_fc_second()


@title('fc layer first')
def run_fc_first():
    @timer('load')
    def load():
        model = Sequential()
        model.add(Dense(8192, input_shape=(7680,)))
        model.add(BatchNormalization(input_shape=(8192,)))
        model.add(Activation('relu', input_shape=(8192,)))

        return model

    test_x = np.random.rand(7680)
    model = load()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


@title('fc layer second')
def run_fc_second():
    @timer('load')
    def load():
        model = Sequential()

        model.add(Dense(8192, input_shape=(8192,)))
        model.add(BatchNormalization(input_shape=(8192,)))
        model.add(Activation('relu', input_shape=(8192,)))

        model.add(Dense(51, input_shape=(8192,)))
        model.add(BatchNormalization(input_shape=(51,)))
        model.add(Activation('softmax', input_shape=(51,)))

        return model

    test_x = np.random.rand(8192)
    model = load()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


if __name__ == '__main__':
    main()
