from keras.layers import Conv2D, Input
from keras.models import Model
import channel
import xy
import numpy as np


X = Input([256, 256, 3])
conv = Conv2D(10, (3, 3))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = channel.forward(data, 10, (3, 3))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (3, 3),
    'stride: ', (1, 1),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (3, 3), strides=(2, 2))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = channel.forward(data, 10, (3, 3), (2, 2))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (3, 3),
    'stride: ', (2, 2),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (3, 3))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (3, 3))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (3, 3),
    'stride: ', (1, 1),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (3, 3), strides=(2, 2))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (3, 3), (2, 2))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (3, 3),
    'stride: ', (2, 2),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (1, 1), strides=(2, 2))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (1, 1), (2, 2))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (1, 1),
    'stride: ', (2, 2),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (1, 1), strides=(3, 3))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (1, 1), (3, 3))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (1, 1),
    'stride: ', (3, 3),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (2, 2), strides=(3, 3))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (2, 2), (3, 3))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (2, 2),
    'stride: ', (3, 3),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (5, 5), strides=(3, 3))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (5, 5), (3, 3))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (5, 5),
    'stride: ', (3, 3),
)

X = Input([256, 256, 3])
conv = Conv2D(10, (4, 4), strides=(3, 3))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (4, 4), (3, 3))
assert model.output_shape[1:] == output.shape[1:], (
    'expected: ', model.output_shape[1:],
    'actual: ', output.shape[1:],
    'kernal: ', (4, 4),
    'stride: ', (3, 3),
)
