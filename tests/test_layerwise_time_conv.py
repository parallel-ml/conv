from models import c3d, vgg16, alexnet, resnet50, xception, yolo
from timer import layerwise
import sys


def test_c3d_original():
    path = 'timer/resource/c3d/conv/c3d'

    sys.stdout = open(path + '.txt', 'w+')
    model = c3d.original(include_fc=False)
    print layerwise.timer(model)


def test_vgg16_original():
    path = 'timer/resource/vgg16/conv/vgg16'

    sys.stdout = open(path + '.txt', 'w+')
    model = vgg16.original(include_fc=False)
    print layerwise.timer(model)


def test_vgg16_channel():
    path = 'timer/resource/vgg16/conv/vgg16'

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = vgg16.channel(include_fc=False)
    print layerwise.timer(model)


def test_vgg16_filter():
    path = 'timer/resource/vgg16/conv/vgg16'

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = vgg16.filter(include_fc=False)
    print layerwise.timer(model)


def test_vgg16_spatial():
    path = 'timer/resource/vgg16/conv/vgg16'

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = vgg16.xy(include_fc=False)
    print layerwise.timer(model)


def test_alexnet_original():
    path = 'timer/resource/alexnet/conv/alexnet'

    sys.stdout = open(path + '.txt', 'w+')
    model = alexnet.original(include_fc=False)
    print layerwise.timer(model)


def test_alexnet_channel():
    path = 'timer/resource/alexnet/conv/alexnet'

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = alexnet.channel(include_fc=False)
    print layerwise.timer(model)


def test_alexnet_filter():
    path = 'timer/resource/alexnet/conv/alexnet'

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = alexnet.filter(include_fc=False)
    print layerwise.timer(model)


def test_alexnet_spatial():
    path = 'timer/resource/alexnet/conv/alexnet'

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = alexnet.xy(include_fc=False)
    print layerwise.timer(model)


def test_resnet50_original():
    path = 'timer/resource/resnet50/conv/resnet50'

    sys.stdout = open(path + '.txt', 'w+')
    model = resnet50.original(include_fc=False)
    print layerwise.timer(model)


def test_resnet50_channel():
    path = 'timer/resource/resnet50/conv/resnet50'

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = resnet50.channel(include_fc=False)
    print layerwise.timer(model)


def test_resnet50_filter():
    path = 'timer/resource/resnet50/conv/resnet50'

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = resnet50.filter(include_fc=False)
    print layerwise.timer(model)


def test_resnet50_spatial():
    path = 'timer/resource/resnet50/conv/resnet50'

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = resnet50.xy(include_fc=False)
    print layerwise.timer(model)


def test_xception_original():
    path = 'timer/resource/xception/conv/xception'

    sys.stdout = open(path + '.txt', 'w+')
    model = xception.original(include_fc=False)
    print layerwise.timer(model)


def test_xception_filter():
    path = 'timer/resource/xception/conv/xception'

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = xception.filter(include_fc=False)
    print layerwise.timer(model)


def test_yolo_orignal():
    path = 'timer/resource/yolo/conv/yolo'

    sys.stdout = open(path + '.txt', 'w+')
    model = yolo.original()
    print layerwise.timer(model)
