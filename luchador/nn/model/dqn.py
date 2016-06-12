from __future__ import absolute_import

from ..core import Model
from ..core import (
    ReLU,
    Dense,
    Conv2D,
    Flatten,
)
from ..core import Normal


def vanilla_dqn(n_actions):

    initializers = {
        'bias': Normal(mean=0.0, stddev=0.01),
        'weight': Normal(mean=0.0, stddev=0.01),
    }

    conv0 = Conv2D(filter_shape=(8, 8), n_filters=32, stride=4,
                   initializers=initializers, padding='SAME')
    conv1 = Conv2D(filter_shape=(4, 4), n_filters=64, stride=2,
                   initializers=initializers)
    conv2 = Conv2D(filter_shape=(3, 3), n_filters=64, stride=1,
                   initializers=initializers)
    dense0 = Dense(n_nodes=512, initializers=initializers)
    dense1 = Dense(n_nodes=n_actions, initializers=initializers)

    model = Model()
    # Note: In original DQN, they have 1 pixel 0 padding for each dimention.
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
