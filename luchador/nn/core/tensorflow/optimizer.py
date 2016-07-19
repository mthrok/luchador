from __future__ import absolute_import

import tensorflow as tf  # nopep8

from .tensor import Operation


class RMSProp(object):
    def __init__(self, learning_rate):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)

    def minimize(self, tensor, var_list=None):
        var_list = [v.tensor for v in var_list]
        op = self.optimizer.minimize(tensor.tensor, var_list=var_list)
        return Operation(op)
