from __future__ import absolute_import

from luchador.nn import (
    Sequential,
    ReLU,
    Dense,
    Conv2D,
    Flatten,
)


def vanilla_dqn(n_actions):
    initializers = {
        'bias': {
            'name': 'Normal',
            'args': {
                'mean': 0.0,
                'stddev': 0.01
            },
        },
        'weight': {
            'name': 'Normal',
            'args': {
                'mean': 0.0,
                'stddev': 0.01,
            },
        },
    }

    conv0 = Conv2D(
        filter_height=8, filter_width=8, n_filters=32, strides=4,
        initializers=initializers, padding='valid')
    conv1 = Conv2D(
        filter_height=4, filter_width=4, n_filters=64, strides=2,
        initializers=initializers, padding='valid')
    conv2 = Conv2D(
        filter_height=3, filter_width=3, n_filters=64, strides=1,
        initializers=initializers, padding='valid')
    dense0 = Dense(n_nodes=512, initializers=initializers)
    dense1 = Dense(n_nodes=n_actions, initializers=initializers)

    model = Sequential()
    model.add_layer(conv0, scope='layer1/conv2D')
    model.add_layer(ReLU(), scope='layer1/ReLU')

    model.add_layer(conv1, scope='layer2/conv2D')
    model.add_layer(ReLU(), scope='layer2/ReLU')

    model.add_layer(conv2, scope='layer3/conv2D')
    model.add_layer(ReLU(), scope='layer3/ReLU')

    model.add_layer(Flatten(), scope='layer4/flatten')

    model.add_layer(dense0, scope='layer5/dense')
    model.add_layer(ReLU(), scope='layer5/ReLU')

    model.add_layer(dense1, scope='layer6/dense')
    return model
