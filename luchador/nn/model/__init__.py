from __future__ import absolute_import

from .preprocessing import image_normalizer

from .dqn import vanilla_dqn


def model_factory(name):
    if name == 'image_normalizer':
        return image_normalizer
    elif name == 'vanilla_dqn':
        return vanilla_dqn
    raise ValueError('Unknown model name: {}'.format(name))
