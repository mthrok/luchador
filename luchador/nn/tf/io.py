import os
import logging

import tensorflow as tf

_LG = logging.getLogger(__name__)


class SummaryWriter(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir

        # Placeholder and session for arbitrary summary creation
        self.summary_ops = {}
        self.placeholder = tf.placeholder(tf.float32, name='summary')

    def init(self, session):
        self.session = session
        self.writer = tf.train.SummaryWriter(
            self.output_dir, session.graph)

    def write(self, summary_str, n_trainings):
        self.writer.add_summary(summary_str, n_trainings)
        self.writer.flush()

    def summarize(self, name, value, type_='scalar'):
        if name not in self.summary_ops:
            if type_ == 'histogram':
                op = tf.histogram_summary(name, self.placeholder)
            elif type_ == 'image':
                op = tf.image_summary(name, self.placeholder)
            else:
                op = tf.scalar_summary(name, self.placeholder)
            self.summary_ops[name] = op
        return self.session.run(
            self.summary_ops[name], feed_dict={self.placeholder: value})


class Saver(object):
    def __init__(self, output_dir, prefix):
        self.prefix = os.path.join(output_dir, prefix)

    def init(self, session):
        self.session = session
        self.saver = tf.train.Saver()

    def save(self, global_step):
        self.saver.save(self.session, self.prefix, global_step)
