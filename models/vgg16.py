from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model
from unit import filter_unit, channel_unit, xy_unit


def original():
    img_input = Input(shape=[224, 224, 3])

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1000, activation='softmax')(x)

    return Model(img_input, x)


def filter():
    img_input = Input(shape=[224, 224, 3])

    # Block 1
    x = filter_unit(img_input, 64, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 64, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = filter_unit(x, 128, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 128, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = filter_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1000, activation='softmax')(x)

    return Model(img_input, x)


def xy():
    img_input = Input(shape=[224, 224, 3])

    # Block 1
    x = xy_unit(img_input, 64, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 64, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = xy_unit(x, 128, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 128, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = xy_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1000, activation='softmax')(x)

    return Model(img_input, x)


def channel():
    img_input = Input(shape=[224, 224, 3])

    # Block 1
    x = channel_unit(img_input, 64, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 64, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = channel_unit(x, 128, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 128, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = channel_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 256, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1000, activation='softmax')(x)

    return Model(img_input, x)
