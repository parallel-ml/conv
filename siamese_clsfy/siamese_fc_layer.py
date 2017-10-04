import numpy as np
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from util.output import title, timer, avg_timer


@title("4 FC layers (4096 nodes per layer)")
def main():
    @timer('load model')
    def load(*args, **kwargs):
        model = Sequential()
        model.add(Dense(4096, input_shape=(7680,)))
        model.add(BatchNormalization(input_shape=(4096,)))
        model.add(Activation('relu', input_shape=(4096,)))

        model.add(Dense(4096, input_shape=(4096,)))
        model.add(BatchNormalization(input_shape=(4096,)))
        model.add(Activation('relu', input_shape=(4096,)))

        model.add(Dense(51, input_shape=(4096,)))
        model.add(BatchNormalization(input_shape=(51,)))
        model.add(Activation('softmax', input_shape=(51,)))
        return model

    model = load()
    test_x = np.random.rand(7680)
    model.predict(np.array([test_x]))

    @avg_timer('inference')
    def forward(*args, **kwargs):
        model.predict(np.array([test_x]))

    forward()

if __name__ == '__main__':
    main()
