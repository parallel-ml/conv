from models import c3d, vgg16, alexnet, resnet50, xception
from timer import layerwise
import sys


def test_c3d():
    path = 'timer/resource/c3d/c3d'

    sys.stdout = open(path + '.txt', 'w+')
    model = c3d.original()
    print layerwise.timer(model)


def test_vgg16():
    path = 'timer/resource/vgg16/vgg16'

    sys.stdout = open(path + '.txt', 'w+')
    model = vgg16.original()
    print layerwise.timer(model)


def test_alexnet():
    path = 'timer/resource/alexnet/alexnet'

    sys.stdout = open(path + '.txt', 'w+')
    model = alexnet.original()
    print layerwise.timer(model)


def test_resnet50():
    path = 'timer/resource/resnet50/resnet50'

    sys.stdout = open(path + '.txt', 'w+')
    model = resnet50.original()
    print layerwise.timer(model)


def test_xception():
    path = 'timer/resource/xception/xception'

    sys.stdout = open(path + '.txt', 'w+')
    model = xception.original()
    print layerwise.timer(model)
