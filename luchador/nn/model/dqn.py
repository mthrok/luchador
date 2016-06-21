from __future__ import absolute_import

from ..core import Model
from ..core import (
    ReLU,
    Dense,
    Conv2D,
    Flatten,
)
from ..core import Normal


def vanilla_dqn(n_actions, data_format):
    initializers = {
        'bias': Normal(mean=0.0, stddev=0.01),
        'weight': Normal(mean=0.0, stddev=0.01),
    }

    conv0 = Conv2D(
        filter_height=8, filter_width=8, n_filters=32, strides=4,
        initializers=initializers, padding='valid', data_format=data_format)
    conv1 = Conv2D(
        filter_height=4, filter_width=4, n_filters=64, strides=2,
        initializers=initializers, padding='valid', data_format=data_format)
    conv2 = Conv2D(
        filter_height=3, filter_width=3, n_filters=64, strides=1,
        initializers=initializers, padding='valid', data_format=data_format)
    dense0 = Dense(n_nodes=512, initializers=initializers)
    dense1 = Dense(n_nodes=n_actions, initializers=initializers)

    model = Model()
    model.add(conv0, scope='layer0/conv2D')
    model.add(ReLU(), scope='layer0/ReLU')

    model.add(conv1, scope='layer1/conv2D')
    model.add(ReLU(), scope='layer1/ReLU')

    model.add(conv2, scope='layer2/conv2D')
    model.add(ReLU(), scope='layer2/ReLU')

    model.add(Flatten(), scope='layer3/flatten')

    model.add(dense0, scope='layer4/dense')
    model.add(ReLU(), scope='layer4/ReLU')

    model.add(dense1, scope='layer5/dense')
    return model
