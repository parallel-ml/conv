import numpy as np
from keras.layers.pooling import MaxPooling1D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model


def main():
    test_x = np.random.rand(64, 256)

    def load(*args, **kwargs):
        N = kwargs['N']

        input = Input(shape=test_x.shape, name='input')

        max1 = MaxPooling1D(pool_size=N, strides=N)(input)
        max2 = MaxPooling1D(pool_size=N / 2, strides=N / 2)(input)
        max3 = MaxPooling1D(pool_size=N / 4, strides=N / 4)(input)
        max4 = MaxPooling1D(pool_size=N / 8, strides=N / 8)(input)

        mrg = Concatenate(axis=1)([max1, max2, max3, max4])

        model = Model(input=input, outputs=mrg)

        print model.summary()

        return model

    model = load(N=64)

    def forward():
        return model.predict(np.array([test_x]))

    test_y = forward()

    print test_y.shape


if __name__ == '__main__':
    main()
