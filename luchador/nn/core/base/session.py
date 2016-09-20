from __future__ import absolute_import

import h5py

__all__ = ['Session']


def _parse_dataset(h5group, prefix=''):
    ret = {}
    for key, value in h5group.items():
        path = '{}/{}'.format(prefix, key) if prefix else key
        if isinstance(value, h5py.Group):
            ret.update(_parse_dataset(value, prefix=path))
        else:
            ret[path] = value
    return ret


class Session(object):
    """Defines common interface for computation handler

    Each backend must implement the following methods:
      - __init__: Construct instance and initialize session
      - run: Run computation
      - initialize: Initialize Variable-s
    """
    def __init__(self):
        raise NotImplementedError(
            '{}.{} is not implemented yet.'
            .format(type(self).__module__, type(self).__name__)
        )

    def run(self, name, inputs, outputs, updates, givens):
        raise NotImplementedError(
            '`run` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )

    def initialize(self):
        raise NotImplementedError(
            '`initialize` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )

    ###########################################################################
    def load_from_file(self, filepath):
        f = h5py.File(filepath, 'r')
        try:
            self.load_dataset(_parse_dataset(f))
        except Exception:
            f.close()
            raise

    def load_dataset(self, dataset):
        raise NotImplementedError(
            '`load_dataset` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )
