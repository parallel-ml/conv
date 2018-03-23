from keras.layers import Conv3D, MaxPooling3D, Input, Flatten, Dense
from keras.models import Model


def original():
    img_input = Input([112, 112, 16, 3])

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    fc = Flatten()(x)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(487)(fc)

    return Model(img_input, fc)
