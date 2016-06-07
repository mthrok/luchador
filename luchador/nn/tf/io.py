import os
import logging
from collections import defaultdict

import tensorflow as tf

_LG = logging.getLogger(__name__)


def _get_summary_func(type_):
    if type_ == 'scalar':
        return tf.scalar_summary
    if type_ == 'image':
        return tf.image_summary
    return tf.histogram_summary


class SummaryWriter(object):
    def __init__(self, output_dir):
        self.summary_ops = defaultdict(list)
        self.output_dir = output_dir

        self.summary_placeholder = tf.placeholder(dtype=tf.float32)

    def init(self, session):
        self.session = session
        self.writer = tf.train.SummaryWriter(self.output_dir, session.graph)

    def register(self, key, type_, tensor):
        func = _get_summary_func(type_)
        self.summary_ops[key].append(func(key, tensor))

    def register_multi(self, key, type_, tensors):
        func = _get_summary_func(type_)
        for tensor in tensors:
            self.summary_ops[key].append(func(tensor.name, tensor))

    def summarize(self, key, global_step, feed_dict={}):
        summaries = self.session.run(
            self.summary_ops[key], feed_dict=feed_dict)
        for summary in summaries:
            self.writer.add_summary(summary, global_step)
        self.writer.flush()


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
