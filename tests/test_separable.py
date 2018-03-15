from keras.layers import Input, SeparableConv2D
from keras.models import Model
from ..utils import separable


def test_original():
    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3))(X)
    model = Model(X, conv)

    conv = separable.original(X, 10, (3, 3))
    test_model = Model(X, conv)
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (1, 1),
    )

    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3), strides=(2, 2))(X)
    model = Model(X, conv)

    conv = separable.original(X, 10, (3, 3), strides=(2, 2))
    test_model = Model(X, conv)
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (2, 2),
    )

    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3), padding='same')(X)
    model = Model(X, conv)

    conv = separable.original(X, 10, (3, 3), padding='same')
    test_model = Model(X, conv)
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (1, 1),
    )

    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3), strides=(2, 2), padding='same')(X)
    model = Model(X, conv)

    conv = separable.original(X, 10, (3, 3), strides=(2, 2), padding='same')
    test_model = Model(X, conv)
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (2, 2),
    )


def test_filter():
    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3))(X)
    model = Model(X, conv)

    conv = separable.filter(X, 10, (3, 3))
    test_model = Model(X, conv)
    print test_model.summary()
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() + 20 == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (1, 1),
    )

    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3), strides=(2, 2))(X)
    model = Model(X, conv)

    conv = separable.filter(X, 10, (3, 3), strides=(2, 2))
    test_model = Model(X, conv)
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() + 20 == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (2, 2),
    )

    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3), padding='same')(X)
    model = Model(X, conv)

    conv = separable.filter(X, 10, (3, 3), padding='same')
    test_model = Model(X, conv)
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() + 20 == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (1, 1),
    )

    X = Input([256, 256, 3])
    conv = SeparableConv2D(10, (3, 3), strides=(2, 2), padding='same')(X)
    model = Model(X, conv)

    conv = separable.filter(X, 10, (3, 3), strides=(2, 2), padding='same')
    test_model = Model(X, conv)
    assert model.output_shape[1:] == test_model.output_shape[1:] \
        and model.count_params() + 20 == test_model.count_params(), (
        'expected: ', model.output_shape[1:], ' param: ', model.count_params(),
        'actual: ', test_model.output_shape[1:], ' param: ', test_model.count_params(),
        'kernal: ', (3, 3),
        'stride: ', (2, 2),
    )