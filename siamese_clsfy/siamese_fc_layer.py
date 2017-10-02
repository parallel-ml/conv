import numpy as np
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from output import title, timer


@title("4 FC layers (4096 nodes per layer)")
def main():
    @timer('load model')
    def load(param):
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

    model = load(None)
    test_x = np.random.rand((7680))

    @timer('inference')
    def forward(param):
        model.predict(np.array([test_x]))

    forward(None)

if __name__ == '__main__':
    main()
