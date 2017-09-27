from keras import model.layers

model = Sequential()
model.add(Dense(4096, input_shape=(7680,)))
model.add(Dense(4096, input_shape=(4096,)))
model.add(Dense(51, input_shape=(4096,)))
