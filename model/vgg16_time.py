from vgg16 import *
import numpy as np
import time


def main():
    start = time.time()
    model = vgg16()
    print 'load time: {:.3f}'.format(time.time() - start)

    test_x = np.random.rand(224, 224, 3)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test_x]))
    print 'inference: {:.3f}'.format((time.time() - start) / 100)


if __name__ == '__main__':
    main()