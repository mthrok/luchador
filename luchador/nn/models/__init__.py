from __future__ import absolute_import

from .preprocessing import image_normalizer

from .dqn import vanilla_dqn


def model_factory(name, **kwargs):
    if name == 'image_normalizer':
        return image_normalizer(**kwargs)
    elif name == 'vanilla_dqn':
        return vanilla_dqn(**kwargs)
    raise ValueError('Unknown model name: {}'.format(name))
