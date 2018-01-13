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
assert model.output_shape[1:] == output.shape[1:], ('expected: ', model.output_shape[1:], 'actual: ', output.shape[1:])

X = Input([256, 256, 3])
conv = Conv2D(10, (3, 3), strides=(2, 2))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = channel.forward(data, 10, (3, 3), (2, 2))
assert model.output_shape[1:] == output.shape[1:], ('expected: ', model.output_shape[1:], 'actual: ', output.shape[1:])

X = Input([256, 256, 3])
conv = Conv2D(10, (3, 3))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (3, 3))
assert model.output_shape[1:] == output.shape[1:], ('expected: ', model.output_shape[1:], 'actual: ', output.shape[1:])

X = Input([256, 256, 3])
conv = Conv2D(10, (3, 3), strides=(2, 2))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (3, 3), (2, 2))
assert model.output_shape[1:] == output.shape[1:], ('expected: ', model.output_shape[1:], 'actual: ', output.shape[1:])

X = Input([256, 256, 3])
conv = Conv2D(10, (5, 5), strides=(2, 2))(X)
model = Model(X, conv)

data = np.random.random_sample([256, 256, 3])
output = xy.forward(data, 10, (5, 5), (2, 2))
assert model.output_shape[1:] == output.shape[1:], ('expected: ', model.output_shape[1:], 'actual: ', output.shape[1:])
