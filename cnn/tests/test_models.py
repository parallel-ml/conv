import sys
from cnn.models import alexnet, vgg16
from keras.utils import plot_model


def test_alexnet():
    path = 'cnn/models/resource/alexnet/alexnet'

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
    path = 'cnn/models/resource/vgg16/vgg16'

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
