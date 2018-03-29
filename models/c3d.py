from keras.layers import Conv3D, MaxPooling3D, Input, Flatten, Dense
from keras.models import Model


NAME = 'c3d'


def original():
    name = NAME + '_original'
    img_input = Input([112, 112, 16, 3])

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name=name + '_1_conv')(img_input)
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name=name + '_2_conv')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name=name + '_3_conv')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name=name + '_4_conv')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name=name + '_5_conv')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name=name + '_6_conv')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name=name + '_7_conv')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name=name + '_8_conv')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    fc = Flatten()(x)
    fc = Dense(4096, activation='relu', name=name + '_9_dense')(fc)
    fc = Dense(4096, activation='relu', name=name + '_10_dense')(fc)
    fc = Dense(487, name=name + '_11_dense')(fc)

    return Model(img_input, fc)
