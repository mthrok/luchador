from __future__ import absolute_import


__all__ = ['Environment']


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
