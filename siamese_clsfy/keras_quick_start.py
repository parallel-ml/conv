import numpy as np
from keras.models import load_model
from output import title, timer

def main():
    optical_flow()
    image()


@title('optical flow (temporal)')
def optical_flow():
    model = None

    @timer('load model')
    def load(path):
        return load_model(path)

    model = load(
        '/home/jiashen/weights/batch_4_noaug/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5')

    test_x = np.random.rand(12, 16, 20)

    # pop the last three layers from training
    for _ in range(3):
        model.pop()

    @timer('inference')
    def forward(param):
        model.predict(np.array([test_x]))

    forward(None)


@title('single frame (spatial)')
def image():
    model = None

    @timer('load model')
    def load(path):
        return load_model(path)

    model = load(
        '/home/jiashen/weights/batch_4_aug/199_epoch-5.2804_loss-0.1080_acc-5.9187_val_loss-0.0662_val_acc.hdf5')

    test_x = np.random.rand(12, 16, 3)

    # pop the last three layers from training
    for _ in range(3):
        model.pop()

    @timer('inference')
    def forward(param):
        model.predict(np.array([test_x]))

    forward(None)


if __name__ == '__main__':
    main()
