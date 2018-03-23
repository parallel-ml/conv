import sys
from ..models import alexnet, vgg16, resnet50, xception, c3d
from keras.utils import plot_model


def test_alexnet():
    path = 'models/resource/alexnet/alexnet'

    sys.stdout = open(path + '.txt', 'w+')
    model = alexnet.original()
    print model.summary()
    plot_model(model, to_file=path + '.png')

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = alexnet.filter()
    print model.summary()
    plot_model(model, to_file=path + '_filter_split.png')

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = alexnet.channel()
    print model.summary()
    plot_model(model, to_file=path + '_channel_split.png')

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = alexnet.xy()
    print model.summary()
    plot_model(model, to_file=path + '_spatial_split.png')


def test_vgg16():
    path = 'models/resource/vgg16/vgg16'

    sys.stdout = open(path + '.txt', 'w+')
    model = vgg16.original()
    print model.summary()
    plot_model(model, to_file=path + '.png')

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = vgg16.filter()
    print model.summary()
    plot_model(model, to_file=path + '_filter_split.png')

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = vgg16.channel()
    print model.summary()
    plot_model(model, to_file=path + '_channel_split.png')

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = vgg16.xy()
    print model.summary()
    plot_model(model, to_file=path + '_spatial_split.png')


def test_resnet50():
    path = 'models/resource/resnet50/resnet50'

    sys.stdout = open(path + '.txt', 'w+')
    model = resnet50.original()
    print model.summary()
    plot_model(model, to_file=path + '.png')

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = resnet50.filter()
    print model.summary()
    plot_model(model, to_file=path + '_filter_split.png')

    sys.stdout = open(path + '_channel_split.txt', 'w+')
    model = resnet50.channel()
    print model.summary()
    plot_model(model, to_file=path + '_channel_split.png')

    sys.stdout = open(path + '_spatial_split.txt', 'w+')
    model = resnet50.xy()
    print model.summary()
    plot_model(model, to_file=path + '_spatial_split.png')


def test_xception():
    path = 'models/resource/xception/xception'

    # sys.stdout = open(path + '.txt', 'w+')
    # model = xception.original()
    # print model.summary()
    # plot_model(model, to_file=path + '.png')

    sys.stdout = open(path + '_filter_split.txt', 'w+')
    model = xception.filter()
    print model.summary()
    plot_model(model, to_file=path + '_filter_split.png')

    # sys.stdout = open(path + '_channel_split.txt', 'w+')
    # model = xception.channel()
    # print model.summary()
    # plot_model(model, to_file=path + '_channel_split.png')
    #
    # sys.stdout = open(path + '_spatial_split.txt', 'w+')
    # model = xception.xy()
    # print model.summary()
    # plot_model(model, to_file=path + '_spatial_split.png')


def test_c3d():
    path = 'models/resource/c3d/c3d'

    sys.stdout = open(path + '.txt', 'w+')
    model = c3d.original()
    print model.summary()
    plot_model(model, to_file=path + '.png')
