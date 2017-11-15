from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Dense(4096, input_shape=(7680,)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(51))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

test_x = np.random.rand(7680)
test_y = np.random.rand(1)

model.fit(test_x, test_y)
