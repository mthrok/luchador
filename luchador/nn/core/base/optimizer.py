from __future__ import absolute_import

__all__ = ['Optimizer']


class Optimizer(object):
    def __init__(self, name):
        self.name = name

    def minimize(self, loss, wrt, **kwargs):
        raise NotImplementedError(
            '`minimize` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def compute_gradients(self, loss, wrt, **kwargs):
        raise NotImplementedError(
            '`compute_gradients` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def apply_gradients(self, loss, grads_and_vars, **kwargs):
        raise NotImplementedError(
            '`apply_gradients` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )
