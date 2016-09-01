from __future__ import absolute_import

from luchador.common import get_subclasses

__all__ = ['Environment', 'get_env']


class Environment(object):
    @property
    def n_actions(self):
        raise NotImplementedError(
            '`n_actions` is not implemented for {}'.format(self.__class__)
        )

    def reset(self):
        raise NotImplementedError(
            '`reset` method is not implemented for {}'.format(self.__class__)
        )

    def step(self, action):
        raise NotImplementedError(
            '`step` method is not implemented for {}'.format(self.__class__)
        )


def get_env(name):
    for Class in get_subclasses(Environment):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Environment: {}'.format(name))
