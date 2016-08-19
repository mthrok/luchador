from __future__ import absolute_import

import logging

from .common import CopyMixin

_LG = logging.getLogger(__name__)

__all__ = ['Initializer']


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
    def export(self):
        args = {}
        for key, value in self.args.items():
            if key == 'seed':
                if (
                        key is None or
                        isinstance(value, int) or
                        isinstance(value, list)
                ):
                    args[key] = value
                    continue

                try:
                    args[key] = value.tolist()
                except AttributeError:
                    _LG.warning('Failed to serialize seed: {}'.format(value))
            else:
                args[key] = value
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
