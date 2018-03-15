"""
    The separable module implements SeparableConv2D layer in Keras
    by using its normal Conv2D layer. Instead of treating it as black
    box, we want to split it internally.
"""
from keras.layers import Conv2D, Lambda, Concatenate, Add, Activation
import keras.backend as K


def original(X, filters, kernal, strides=(1, 1), padding='valid', activation='relu'):
    """
        We assume this is the original implementation by Xception author.
        We have # channel kernal run depth wise convolution and concatenate
        output to a single tensor. Then run a spatial wise convolution based
        on the filter #.
    """
    channel = K.int_shape(X)[-1]
    X = Lambda(lambda x: [x[:, :, :, k:k+1] for k in range(channel)])(X)
    X = [Conv2D(1, kernal, strides=strides, padding=padding, use_bias=False)(x) for x in X]
    X = Concatenate(axis=-1)(X)
    X = Conv2D(filters, (1, 1), padding=padding, activation=activation)(X)
    return X


def filter(X, filters, kernal, strides=(1, 1), padding='valid', activation='relu'):
    """
        For filter function, the only difference is we do not concatenate the
        output from first Conv2D layer. Instead, it will run parallel spatial
        convolution and add together.
    """
    channel = K.int_shape(X)[-1]
    X = Lambda(lambda x: [x[:, :, :, k:k + 1] for k in range(channel)])(X)
    X = [Conv2D(1, kernal, strides=strides, padding=padding, use_bias=False)(x) for x in X]
    X = [Conv2D(filters, (1, 1), padding=padding)(x) for x in X]
    X = Add()(X)
    X = Activation(activation)(X)
    return X
