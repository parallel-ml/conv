"""
    This module calculates layer-wise time for selected model.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv3D
import time


def timer(model):
    result = ''
    for layer in model.layers:
        if layer.__class__.__name__ == 'InputLayer' \
                or layer.__class__.__name__ == 'Flatten' \
                or layer.__class__.__name__ == 'MaxPooling3D':
            continue

        input_shape = list(layer.input_shape)[1:]
        data = np.random.random_sample(input_shape)
        input = InputLayer(input_shape)

        config = layer.get_config()

        if layer.__class__.__name__ == 'Dense':
            _layer = Dense.from_config(config)
        else:
            _layer = Conv3D.from_config(config)

        _model = Sequential()
        _model.add(input)
        _model.add(_layer)

        start = time.time()
        for _ in range(50):
            _model.predict(np.array([data]))
        result += '{:<10s}: {:.2f} s\n'.format(layer.name, (time.time() - start) / 50)
    return result
