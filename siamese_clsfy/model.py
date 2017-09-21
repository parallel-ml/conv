from keras.layers.convolutional import Conv2D
from keras.models import load_model
# from keras.activations import Dense, Activation
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from memory_profiler import profile


# input_dim_ordering should be set to "tf"
@profile
def get_model(nb_class=1000, bias=True, act='relu', bn=True, dropout=False, moredense=False, nb_filter=256,
              nb_channel=3, clsfy_act='softmax'):
    # input image is 16x12, RGB(x3)
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), use_bias=bias, padding='same', input_shape=(12, 16, nb_channel)))
    # now model.output_shape == (None, 12, 16, 128)
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_1'))

    model.add(Conv2D(nb_filter, (3, 3), use_bias=bias, padding='same'))
    # now model.output_shape == (None, 12, 16, 128)
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_2'))

    model.add(Conv2D(nb_filter, (3, 3), use_bias=bias, padding='same'))
    # now model.output_shape == (None, 12, 16, 128)
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_3'))

    model.add(Flatten())
    # now model.output_shape == (None, 24576)
    model.add(Dense(256, input_dim=24576, use_bias=bias, name='dense_4'))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_4'))
    if dropout:
        model.add(Dropout(0.5))

    if moredense:
        model.add(Dense(256, use_bias=bias, name='dense_deep'))
        if bn:
            model.add(BatchNormalization())
        model.add(Activation(act, name='relu_deep'))
        if dropout:
            model.add(Dropout(0.5))

    # it needs to be detached later video activity learning, but for the first image classification, we use it.
    model.add(Dense(nb_class, use_bias=bias, name='dense_5'))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(clsfy_act, name='activation_5'))

    return model


if __name__ == '__main__':
    model = load_model('/home/jiashen/weights/0000_epoch-4.0079_loss-0.0253_acc-4.1435_val_loss-0.0266_val_acc.hdf5')
