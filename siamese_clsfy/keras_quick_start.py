from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import time

model = load_model('/home/jiashen/weights/clsfybatch_4/0000_epoch-4.0079_loss-0.0253_acc-4.1435_val_loss-0.0266_val_acc.hdf5')

test_x = np.random.rand(8192)

# pop the last three layers from training 
for _ in range(3):
    model.pop()

start = time.time()
model.predict(np.array([test_x]))
print 'FC layer {:.3f}s'.format(time.time() - start)
