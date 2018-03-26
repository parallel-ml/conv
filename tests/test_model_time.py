from ..models import alexnet, vgg16
import numpy as np
import time


def test_vgg16_channel():
    model = vgg16.channel()
    image = np.random.random_sample((224, 224, 3))
    start = time.time()
    for _ in range(20):
        model.predict(np.array([image]))
    print 'vgg16 channel split:', (time.time() - start) / 20, 's'


def test_vgg16_spatial():
    model = vgg16.xy()
    image = np.random.random_sample((224, 224, 3))
    start = time.time()
    for _ in range(20):
        model.predict(np.array([image]))
    print 'vgg16 spatial split:', (time.time() - start) / 20, 's'


def test_vgg16_filter():
    model = vgg16.filter()
    image = np.random.random_sample((224, 224, 3))
    start = time.time()
    for _ in range(20):
        model.predict(np.array([image]))
    print 'vgg16 filters split:', (time.time() - start) / 20, 's'


def test_vgg16_original():
    model = vgg16.original()
    image = np.random.random_sample((224, 224, 3))
    start = time.time()
    for _ in range(20):
        model.predict(np.array([image]))
    print 'vgg16 original:', (time.time() - start) / 20, 's'
