import numpy as np
from keras.layers import MaxPooling1D, Dense, Activation, BatchNormalization, Input
from keras.layers.merge import Concatenate
from keras.models import load_model, Model, Sequential
from memory_profiler import profile

from util.output import title, timer, avg_timer, subtitle

TIMING = False


@title('maxpooling layer')
def main():
    optical_flow()
    image()


def load_helper(mode):
    if mode == 'optical flow':
        return load(
            path='/home/jiashen/weights/batch_4_noaug/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5')
    elif mode == 'image':
        return load(
            path='/home/jiashen/weights/batch_4_aug/199_epoch-5.2804_loss-0.1080_acc-5.9187_val_loss-0.0662_val_acc.hdf5')
    elif mode == 'maxpool':
        N = 100
        while N <= 2000:
            @subtitle(str(N) +  ' image frames')
            @profile
            def loadmax(N):
                return maxpool(N=N)

            loadmax(N)

            N += 100
    elif mode == 'fc':
        return fc()


@timer('load model', timing=TIMING)
def load(*args, **kwargs):
    return load_model(kwargs['path'])


def fc(*args, **kwargs):
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


def maxpool(*args, **kwargs):
    N = kwargs['N']

    input = Input(shape=(N, 256), name='input')

    max1 = MaxPooling1D(pool_size=N, strides=N)(input)
    max2 = MaxPooling1D(pool_size=N / 2, strides=N / 2)(input)
    max3 = MaxPooling1D(pool_size=N / 4, strides=N / 4)(input)
    max4 = MaxPooling1D(pool_size=N / 8, strides=N / 8)(input)

    mrg = Concatenate(axis=1)([max1, max2, max3, max4])

    model = Model(input=input, outputs=mrg)

    return model


@avg_timer('inference', timing=TIMING)
def forward(*args, **kwargs):
    model = kwargs['model']
    test_x = kwargs['test_x']
    model.predict(np.array([test_x]))


@title('optical flow (temporal)')
def optical_flow():
    model = load(
        path='/home/jiashen/weights/batch_4_noaug/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5')

    test_x = np.random.rand(12, 16, 20)

    # pop the last three layers from training
    for _ in range(3):
        model.pop()

    output = model.predict(np.array([test_x]))
    print output.shape

    forward(model=model, test_x=test_x)


@title('single frame (spatial)')
def image():
    model = load(
        path='/home/jiashen/weights/batch_4_aug/199_epoch-5.2804_loss-0.1080_acc-5.9187_val_loss-0.0662_val_acc.hdf5')

    test_x = np.random.rand(12, 16, 3)

    # pop the last three layers from training
    for _ in range(3):
        model.pop()

    output = model.predict(np.array([test_x]))
    print output.shape

    forward(model=model, test_x=test_x)


if __name__ == '__main__':
    main()
