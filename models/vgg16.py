from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model
from unit import filter_unit, channel_unit, xy_unit


NAME = 'vgg16'


def original(include_fc=True):
    name = NAME + '_original'
    img_input = Input(shape=[220, 220, 3])

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name=name + '_1_conv')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name=name + '_2_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name=name + '_3_conv')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name=name + '_4_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name=name + '_5_conv')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name=name + '_6_conv')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name=name + '_7_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name=name + '_8_conv')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name=name + '_9_conv')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name=name + '_10_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name=name + '_11_conv')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name=name + '_12_conv')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name=name + '_13_conv')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    if include_fc:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name=name + '_14_dense')(x)
        x = Dense(4096, activation='relu', name=name + '_15_dense')(x)
        x = Dense(1000, activation='softmax', name=name + '_16_dense')(x)

    return Model(img_input, x)


def filter(include_fc=True):
    name = NAME + '_filter'
    img_input = Input(shape=[220, 220, 3])

    # Block 1
    x = filter_unit(img_input, 64, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_1')
    x = filter_unit(x, 64, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_2')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = filter_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_3')
    x = filter_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_4')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = filter_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_5')
    x = filter_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_6')
    x = filter_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_7')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_8')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_9')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_10')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_11')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_12')
    x = filter_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_13')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    if include_fc:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name=name + '_14_dense')(x)
        x = Dense(4096, activation='relu', name=name + '_15_dense')(x)
        x = Dense(1000, activation='softmax', name=name + '_16_dense')(x)

    return Model(img_input, x)


def xy(include_fc=True):
    name = NAME + '_spatial'
    img_input = Input(shape=[220, 220, 3])

    # Block 1
    x = xy_unit(img_input, 64, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_1')
    x = xy_unit(x, 64, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_2')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = xy_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_3')
    x = xy_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_4')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = xy_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_5')
    x = xy_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_6')
    x = xy_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_7')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_8')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_9')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_10')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_11')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_12')
    x = xy_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_13')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    if include_fc:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name=name + '_14_dense')(x)
        x = Dense(4096, activation='relu', name=name + '_15_dense')(x)
        x = Dense(1000, activation='softmax', name=name + '_16_dense')(x)

    return Model(img_input, x)


def channel(include_fc=True):
    name = NAME + '_channel'
    img_input = Input(shape=[220, 220, 3])

    # Block 1
    x = channel_unit(img_input, 64, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_1')
    x = channel_unit(x, 64, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_2')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = channel_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_3')
    x = channel_unit(x, 128, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_4')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = channel_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_5')
    x = channel_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_6')
    x = channel_unit(x, 256, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_7')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_8')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_9')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_10')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_11')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_12')
    x = channel_unit(x, 512, (3, 3), max_pooling=False, padding='same', activation='relu', name=name + '_13')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    if include_fc:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name=name + '_14_dense')(x)
        x = Dense(4096, activation='relu', name=name + '_15_dense')(x)
        x = Dense(1000, activation='softmax', name=name + '_16_dense')(x)

    return Model(img_input, x)
