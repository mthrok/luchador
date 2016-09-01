from __future__ import absolute_import

import logging

from .common import (
    CopyMixin,
    get_subclasses,
)

_LG = logging.getLogger(__name__)

__all__ = [
    'Initializer', 'get_initializer',
]


class Initializer(CopyMixin, object):
    """Common interface (copy and export methods) for Initializer classes"""
    def __init__(self, **kwargs):
        """Validate and store arguments passed to subclass __init__ method

        As these **args are used to create a copy of instance, arguments which
        cannot be passed to constructor should not be passed. In other way, the
        signature of this constructor must match to the signature of the
        constractor of subclass object being created.
        """
        super(Initializer, self).__init__()
        self._store_args(kwargs)

    ###########################################################################
    def _export_seed(self, args, key, value):
        if (
                value is None or
                isinstance(value, int) or
                isinstance(value, list)
        ):
            args[key] = value
            return

        try:
            args[key] = value.tolist()
        except AttributeError:
            _LG.warning('Failed to serialize seed: {}'.format(value))

    def export(self):
        args = {}
        for key, value in self.args.items():
            if key == 'seed':
                self._export_seed(args, key, value)
                continue

            args[key] = value.export() if hasattr(value, 'export') else value
        return {
            'name': self.__class__.__name__,
            'args': args
        }

    ###########################################################################
    def __call__(self, shape):
        self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError(
            '`sample` method is not implemented for {}.{}'
            .format(type(self).__module__, type(self).__name__))


def get_initializer(name):
    for Class in get_subclasses(Initializer):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Initializer: {}'.format(name))
