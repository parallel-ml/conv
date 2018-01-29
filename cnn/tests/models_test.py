import sys
from cnn.models import alexnet
from keras.utils import plot_model

path = 'cnn/models/resource/'

sys.stdout = open(path + 'alexnet.txt', 'w+')
model = alexnet.original()
print model.summary()
plot_model(model, to_file=path + 'alexnet.png')

sys.stdout = open(path + 'alexnet_filter_split.txt', 'w+')
model = alexnet.filter()
print model.summary()
plot_model(model, to_file=path + 'alexnet_filter_split.png')

sys.stdout = open(path + 'alexnet_channel_split.txt', 'w+')
model = alexnet.channel()
print model.summary()
plot_model(model, to_file=path + 'alexnet_channel_split.png')

sys.stdout = open(path + 'alexnet_spatial_split.txt', 'w+')
model = alexnet.xy()
print model.summary()
plot_model(model, to_file=path + 'alexnet_spatial_split.png')

