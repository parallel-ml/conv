from models import c3d, vgg16, alexnet, resnet50, xception, yolo
from timer import layerwise
import sys


def test_c3d():
    path = 'timer/resource/c3d/c3d_conv'

    sys.stdout = open(path + '.txt', 'w+')
    model = c3d.original(include_fc=False)
    print layerwise.timer(model)


def test_vgg16():
    path = 'timer/resource/vgg16/vgg16_conv'

    sys.stdout = open(path + '.txt', 'w+')
    model = vgg16.original(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = vgg16.channel(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = vgg16.filter(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = vgg16.xy(include_fc=False)
    print layerwise.timer(model)


def test_alexnet():
    path = 'timer/resource/alexnet/alexnet_conv'

    sys.stdout = open(path + '.txt', 'w+')
    model = alexnet.original(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = alexnet.channel(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = alexnet.filter(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = alexnet.xy(include_fc=False)
    print layerwise.timer(model)


def test_resnet50():
    path = 'timer/resource/resnet50/resnet50_conv'

    sys.stdout = open(path + '.txt', 'w+')
    model = resnet50.original(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = resnet50.channel(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = resnet50.filter(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = resnet50.xy(include_fc=False)
    print layerwise.timer(model)


def test_xception():
    path = 'timer/resource/xception/xception_conv'

    sys.stdout = open(path + '.txt', 'w+')
    model = xception.original(include_fc=False)
    print layerwise.timer(model)

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = xception.filter(include_fc=False)
    print layerwise.timer(model)


def test_yolo():
    path = 'timer/resource/yolo/yolo_conv'

    sys.stdout = open(path + '.txt', 'w+')
    model = yolo.original()
    print layerwise.timer(model)
