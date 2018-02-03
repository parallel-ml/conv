from vgg16 import channel as channel_model
import keras
import argparse


ROOT = 'cnn/models/weights/'
DENSE = 19


def main():
    vgg16 = keras.applications.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                                     pooling=None, classes=1000)
    channel(vgg16, args.num)


def channel(model, num):
    """
        Take care of channel split weights input.

        Args:
            model: Pre-trained vgg16 model.
            num: Number of partition.
    """
    vgg16_channel = channel_model()
    num_layer = 0
    for i, layer in enumerate(model.layers):
        # make not dense layer
        if len(layer.get_weights()) > 0 and i <= DENSE:
            # offset the lambda layer
            num_layer += 1
            weights = channel_weights(layer.get_weights(), num)
            for weight in weights:
                channel_layer = vgg16_channel.layers[num_layer]
                channel_layer.set_weights(weight)
                num_layer += 1
            # offset the concatenate layer
            num_layer += 1
        elif i <= DENSE:
            # offset the layer without weights
            num_layer += 1
        else:
            channel_layer = vgg16_channel.layers[num_layer]
            channel_layer.set_weights(layer.get_weights())
            num_layer += 1
    vgg16_channel.save(ROOT + 'vgg16_channel_split.h5')


def channel_weights(weights, num):
    """
        Take care of each weight partition

        Args:
            weights: A numpy array with [kernal, bias].
            num: Number of partition.

        Returns:
            A list of weights. Each cell is comprised of a length two list.
            The first index is kernal weight, and the second is bias, which
            works naturally with Keras set_weights function.

            [[Conv1 kernal, Conv1 bias], [Conv2 kernal, Conv2 bias] ... ]
    """
    kernal, bias = weights[0], weights[1]
    filters = bias.shape[-1]
    size, last = filters / num, filters - filters / num * (num - 1)
    split_weights = []
    for i in range(num):
        if i != num - 1:
            lb, rb = i * size, (i + 1) * size
        else:
            lb, rb = filters - last, filters
        split_weights.append([kernal[:, :, :, lb:rb], bias[lb:rb]])
    return split_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run extractor to get weights from pre-trained ImageNet model.'
                                                 'Partition weights according to number of division')
    parser.add_argument('--num', dest='num', default=3, type=int, help='Number of partition.')
    args = parser.parse_args()
    main()
