from __future__ import absolute_import

import os
import errno

import h5py

__all__ = ['Saver']


class Saver(object):
    # Add functionalities similart to tensorflow.Saver
    def __init__(self, filepath):
        dirname = os.path.dirname(filepath)
        if dirname:
            try:
                os.makedirs(dirname)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

        self.filepath = filepath

    def save(self, data):
        f = h5py.File(self.filepath, 'a')
        for key, value in data.items():
            if key in f:
                del f[key]
            f.create_dataset(key, data=value, chunks=True)
        f.flush()
        f.close()
