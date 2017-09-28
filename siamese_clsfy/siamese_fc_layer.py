from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from memory_profiler import profile
import numpy as np
import time

@profile
def main():
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
    
    test_x = np.random.rand((7680))
    
    start = time.time() 
    model.predict(np.array([test_x]))
    print 'simaese fc layer inference {:.3f}'.format(time.time() - start)



if __name__ == '__main__':
    main()
