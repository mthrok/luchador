from __future__ import absolute_import

import warnings

import tensorflow as tf  # nopep8

from ..base import Optimizer as BaseOptimizer
from .tensor import Operation


class TFOptimizer(BaseOptimizer):
    def minimize(self, loss, wrt=None, **kwargs):
        loss, var_list = loss.tensor, [v.tensor for v in wrt]
        op = self.optimizer.minimize(loss, var_list=var_list, **kwargs)
        return Operation(op)

    def compute_gradients(self, loss, wrt, **kwargs):
        return self.optimizer.compute_gradients(loss, var_list=wrt, **kwargs)

    def apply_gradients(self, loss, grads_and_vars, **kwargs):
        return self.optimizer.apply_gradients(grads_and_vars, **kwargs)


class RMSProp(TFOptimizer):
    def __init__(self, learning_rate,
                 decay1=0.0, decay2=0.95,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        if decay1:
            warnings.warn(
                'Non-zero value was given to `decay1` parameter. '
                'Notice the mplementation difference of RMSProps in '
                'Theano backend and Tensorflow backend, which may '
                'cause different results.',
                RuntimeWarning
            )
        super(RMSProp, self).__init__(name)
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=decay2, momentum=decay1, **kwargs)
