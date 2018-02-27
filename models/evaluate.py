from keras.models import load_model
import numpy as np
from os.path import expanduser
import argparse
import glob
import cv2

HOME = expanduser("~")
IMAGE_DIR = ''
LABEL = ''


def vgg16_channel_split():
    model = load_model(HOME + '/weights/' + 'vgg16_channel_split.h5')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    images = []
    for filename in glob.glob(IMAGE_DIR + '*.JPEG'):
        im = cv2.imread(filename)
        images.append(im)
    label_file = open(LABEL)
    label_text = label_file.read().splitlines()
    labels = []
    for lb in label_text:
        label = np.zeros(1000)
        label[int(lb)] = 1
        labels.append(label)
    model.evaluate(np.array(images), np.array(labels))


def main():
    vgg16_channel_split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pre-trained split model on Image Net.')
    parser.add_argument('--val', dest='val', required=True, help='validation set directory.')
    parser.add_argument('--label', dest='label', required=True, help='label file path.')
    args = parser.parse_args()
    IMAGE_DIR = args.val
    LABEL = args.label
    main()
