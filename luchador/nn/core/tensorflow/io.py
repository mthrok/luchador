from __future__ import absolute_import

import os
import logging

import tensorflow as tf

_LG = logging.getLogger(__name__)

__all__ = ['Saver']


class Saver(object):
    def __init__(
            self, output_dir, prefix, max_to_keep=5, checkpoint_interval=1):
        self.prefix = os.path.join(output_dir, prefix)
        self.checkpoint_interval = checkpoint_interval
        self.max_to_keep = max_to_keep

    def init(self, session):
        self.session = session
        self.saver = tf.train.Saver(
            max_to_keep=self.max_to_keep,
            keep_checkpoint_every_n_hours=self.checkpoint_interval
        )

    def save(self, global_step):
        self.saver.save(self.session, self.prefix, global_step)
