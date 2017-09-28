from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import time
from memory_profiler import profile

@profile
def main():
    model = load_model('/home/jiashen/weights/batch_4_noaug/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5')

    test_x = np.random.rand(12, 16, 20)

    # pop the last three layers from training 
    for _ in range(3):
        model.pop()

    start = time.time()
    model.predict(np.array([test_x]))
    print 'image {:.3f}s'.format(time.time() - start)


if __name__ == '__main__':
    print '+++++++++++++++++++++optical flow++++++++++++++++++++++++++++'
    main()
