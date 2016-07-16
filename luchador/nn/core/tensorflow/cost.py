from __future__ import absolute_import

import tensorflow as tf

from ..core import (
    SSE as BaseSSE,
)

__all__ = ['SSE']


def _clip_delta(delta, min_delta, max_delta):
    with tf.name_scope('clip_delta'):
        if max_delta:
            with tf.name_scope('max'):
                delta = tf.minimum(delta, max_delta, name='clip_by_delta_max')
        if min_delta:
            with tf.name_scope('min'):
                delta = tf.maximum(delta, min_delta, name='clip_by_delta_min')
    return delta


class SSE(BaseSSE):
    def build(self, target, prediction):
        with tf.name_scope('SSE'):
            delta = tf.sub(target, prediction, 'delta')
            delta = _clip_delta(delta, self.min_delta, self.max_delta)
            err = tf.square(delta, name='squared_delta')
            err = tf.reduce_sum(err, reduction_indices=1, name='SSE')
            err = tf.reduce_mean(err, name='SSE_over_batch')
            self.target, self.prediction, self.error = target, prediction, err
            return err
