from __future__ import absolute_import

from luchador.nn import (
    Model,
    ReLU,
    Dense,
    Conv2D,
    Flatten,
    Normal
)


def vanilla_dqn(n_actions):
    initializers = {
        'bias': Normal(mean=0.0, stddev=0.01),
        'weight': Normal(mean=0.0, stddev=0.01),
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

    model = Model()
    model.add_layer(conv0, scope='layer0/conv2D')
    model.add_layer(ReLU(), scope='layer0/ReLU')

    model.add_layer(conv1, scope='layer1/conv2D')
    model.add_layer(ReLU(), scope='layer1/ReLU')

    model.add_layer(conv2, scope='layer2/conv2D')
    model.add_layer(ReLU(), scope='layer2/ReLU')

    model.add_layer(Flatten(), scope='layer3/flatten')

    model.add_layer(dense0, scope='layer4/dense')
    model.add_layer(ReLU(), scope='layer4/ReLU')

    model.add_layer(dense1, scope='layer5/dense')
    return model
