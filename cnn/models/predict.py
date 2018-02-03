from keras.models import load_model
import numpy as np


ROOT = 'cnn/models/weights/'


def vgg16_channel_split():
    model = load_model(ROOT + 'vgg16_channel_split.h5')
    image = np.random.random_sample([224, 224, 3])
    output = model.predict(np.array([image]))
    print model.summary()
    print output
