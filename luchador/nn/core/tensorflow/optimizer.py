from __future__ import absolute_import

import warnings

import tensorflow as tf  # nopep8

from ..base import Optimizer as BaseOptimizer
from .tensor import Operation


class TFOptimizer(BaseOptimizer):
    def _parse_kwargs(self, kwargs):
        keys_and_defs1 = [
            ('gate_gradients', 1),
            ('aggregation_method', None),
            ('colocate_gradients_with_ops', False),
            ('grad_loss', None)
        ]
        keys_and_defs2 = [
            ('global_step', None),
            ('name', None)
        ]
        kws_compute_gradients = {
            key: kwargs.get(key, default_value)
            for key, default_value in keys_and_defs1}
        kws_apply_gradients = {
            key: kwargs.get(key, default_value)
            for key, default_value in keys_and_defs2}
        return [kws_compute_gradients, kws_apply_gradients]

    def minimize(self, loss, wrt=None, **kwargs):
        kws1, kws2 = self._parse_kwargs(kwargs)
        grads_and_vars = self.compute_gradients(loss, wrt=wrt, **kws1)
        return self.apply_gradients(grads_and_vars, **kws2)

    def compute_gradients(self, loss, wrt, **kwargs):
        loss = loss.tensor
        # TODO: Add support for single tensor
        var_list = [v.tensor for v in wrt] if wrt else None
        # TODO: Wrap this with Tensor class
        grads_and_vars = self.optimizer.compute_gradients(
            loss, var_list=var_list, **kwargs)
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, **kwargs):
        # TODO: Add parser once compute_gradients wraps `grads_and_vars`
        minimize_op = self.optimizer.apply_gradients(grads_and_vars, **kwargs)
        return Operation(minimize_op)


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
