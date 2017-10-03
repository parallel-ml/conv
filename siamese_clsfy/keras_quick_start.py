import numpy as np
from keras.models import load_model

from util.output import title, timer


def main():
    optical_flow()
    image()


@timer('load model')
def load(*args, **kwargs):
    return load_model(kwargs['path'])


@timer('inference')
def forward(*args, **kwargs):
    model = kwargs['model']
    test_x = kwargs['test_x']
    model.predict(np.array([test_x]))


@title('optical flow (temporal)')
def optical_flow():
    model = load(
        path='/home/jiashen/weights/batch_4_noaug/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5')

    test_x = np.random.rand(12, 16, 20)

    # pop the last three layers from training
    for _ in range(3):
        model.pop()

    forward(model=model, test_x=test_x)


@title('single frame (spatial)')
def image():
    model = load(
        path='/home/jiashen/weights/batch_4_aug/199_epoch-5.2804_loss-0.1080_acc-5.9187_val_loss-0.0662_val_acc.hdf5')

    test_x = np.random.rand(12, 16, 3)

    # pop the last three layers from training
    for _ in range(3):
        model.pop()

    forward(model=model, test_x=test_x)


if __name__ == '__main__':
    main()
