from __future__ import absolute_import

import os
import h5py


class Saver(object):
    def __init__(self, output_dir, prefix, max_to_keep, checkpoint_interval=1):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_to_keep = max_to_keep
        self.checkpoint_interval = checkpoint_interval

    def save(self, **data):
        # Add checkpointing and max keeping
        path = os.path.join(self.output_dir, self.prefix)
        f = h5py.File(path)
        for key, value in data.items():
            if key not in f:
                f.create_dataset(key, data=value, chunks=True)
            else:
                f[key] = value
        f.flush()
        f.close()

    def restore(self, path):
        f = h5py.File(path, 'r')
        for key in f:
            value = f[key]
            # TODO: Set value here
            # TODO: Add test for io
