"""
    This module calculates layer-wise time for selected model.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv3D, Conv2D, SeparableConv2D
import time


def timer(model):
    result = ''
    for layer in model.layers:
        layer_name = layer.__class__.__name__
        if layer_name == 'InputLayer' or layer_name == 'Flatten' \
                or layer_name == 'MaxPooling3D' or layer_name == 'Activation' \
                or layer_name == 'MaxPooling2D' or layer_name == 'ZeroPadding2D' \
                or layer_name == 'Add' or layer_name == 'AveragePooling2D' \
                or layer_name == 'GlobalAveragePooling2D':
            continue

        input_shape = list(layer.input_shape)[1:]
        data = np.random.random_sample(input_shape)
        input = InputLayer(input_shape)

        config = layer.get_config()

        if layer_name == 'Dense':
            _layer = Dense.from_config(config)
        elif layer_name == 'Conv3D':
            _layer = Conv3D.from_config(config)
        elif layer_name == 'Conv2D':
            _layer = Conv2D.from_config(config)
        elif layer_name == 'SeparableConv2D':
            _layer = SeparableConv2D.from_config(config)
        else:
            raise Exception(layer_name, ' layer name not included')

        _model = Sequential()
        _model.add(input)
        _model.add(_layer)

        start = time.time()
        for _ in range(50):
            _model.predict(np.array([data]))
        result += '{:>20s}: {:.3f} s\n'.format(layer.name, (time.time() - start) / 50)
    return result
