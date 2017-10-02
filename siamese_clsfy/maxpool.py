import numpy as np
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential
from output import title, timer

def main():
    def load(param, N):
        model = Sequential()
        model.add(GlobalAveragePooling2D(data_format=(N, 256)))
        model.add(MaxPooling2D(pool_size = (N / 2, 256), strides=N / 2))
        model.add(MaxPooling2D(pool_size=(N / 4, 256), strides=N / 4))
        model.add(MaxPooling2D(pool_size=(N / 8, 256), strides=N / 8))
        return model

    model = load(None, 64)
    test_x = np.random.rand((64, 256))

    def forward(param):
        model.predict(np.array([test_x]))

    forward(None)

if __name__ == '__main__':
    main()