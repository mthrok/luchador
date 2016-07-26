from __future__ import absolute_import

import logging

import tensorflow as tf

_LG = logging.getLogger(__name__)


class _SummaryOperations(object):
    """Create placeholder and summary operations for the given tensors"""
    def __init__(self, summary_type, tensors):
        self._build_operations(summary_type, tensors)

    def _build_operations(self, summary_type, tensors):
        _summary_func = {
            'scalar': tf.scalar_summary,
            'image': tf.image_summary,
            'audio': tf.audio_summary,
            'histogram': tf.histogram_summary,
        }
        summary_fn = _summary_func[summary_type]

        self.ops, self.feed_keys = [], []
        for name, tensor in tensors.items():
            shape, dtype = tensor.shape, 'float32'  # tensor.dtype
            placeholder = tf.placeholder(dtype, shape=shape)
            self.feed_keys.append(placeholder)
            self.ops.append(summary_fn(name, placeholder))

    def get_feed_dict(self, feed_values):
        return {key: val for key, val in zip(self.feed_keys, feed_values)}


class SummaryWriter(object):
    def __init__(self, output_dir, graph=None):
        self.output_dir = output_dir

        self.summary_ops = {}
        self.writer = tf.train.SummaryWriter(self.output_dir)

    def init(self, **kwargs):
        pass

    def add_graph(self, graph=None):
        if graph:
            self.writer.add_graph(graph)

    def register(self, key, summary_type, tensors):
        self.summary_ops[key] = _SummaryOperations(summary_type, tensors)

    def summarize(self, key, global_step, values):
        cfg = self.summary_ops[key]
        feed_dict = cfg.get_feed_dict(values)
        with tf.Session() as session:
            summaries = session.run(cfg.ops, feed_dict=feed_dict)
            for summary in summaries:
                self.writer.add_summary(summary, global_step)
            self.writer.flush()
