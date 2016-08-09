from __future__ import absolute_import

import tensorflow as tf  # nopep8

from ..base import Optimizer as BaseOptimizer
from .tensor import Operation

__all__ = ['RMSProp', 'GravesRMSProp']


class TFOptimizer(BaseOptimizer):
    def _parse_kwargs(self, kwargs):
        keys_and_defaults1 = [
            ('gate_gradients', 1),
            ('aggregation_method', None),
            ('colocate_gradients_with_ops', False),
            ('grad_loss', None)
        ]
        keys_and_defaults2 = [
            ('global_step', None),
            ('name', None)
        ]
        kws_compute_gradients = {
            key: kwargs.get(key, default_value)
            for key, default_value in keys_and_defaults1}
        kws_apply_gradients = {
            key: kwargs.get(key, default_value)
            for key, default_value in keys_and_defaults2}
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
                 decay=0.95, momentum=None,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        super(RMSProp, self).__init__(name)
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=decay, momentum=momentum,
            epsilon=epsilon, **kwargs)


class GravesRMSProp(TFOptimizer):
    """Implements """
    def __init__(self, learning_rate,
                 decay1=0.0, decay2=0.95, epsilon=1e-2,
                 name='GravesRMSProp', **kwargs):
        # TODO: Add support for momentum
        super(GravesRMSProp, self).__init__(name)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate, name=name)
        self.decay1 = decay1
        self.decay2 = decay2
        self.epsilon = epsilon

    def apply_gradient(self, grads_and_vars, **kwargs):
        # TODO: Add parser once compute_gradients wraps `grads_and_vars`
        # TODO: Save intermediate Variables in slot
        mean_grads1, mean_grads1_updates = [], []
        mean_grads2, mean_grads2_updates = [], []
        new_grads_and_vars = [], []
        d1, d2, ep = self.decay1, self.decay2, self.epsilon
        for grad, var in grads_and_vars:
            with tf.variable_scope(self.name):
                name = '{}_mean'.format(grad.name)
                mean_grad1 = tf.get_variable(
                    name=name, shape=grad.shape, dtype=grad.dtype,
                    initializer=tf.constatnt_initializer(0))

                name = '{}_squared_mean'.format(grad.name)
                mean_grad2 = tf.get_variable(
                    name=name, shape=grad.shape, dtype=grad.dtype,
                    initializer=tf.constatnt_initializer(0))

                mean_grad1_ = d1 * mean_grad1 + (1.0 - d1) * grad
                mean_grad2_ = d2 * mean_grad2 + (1.0 - d2) * tf.square(grad)

                rms = tf.sqrt(mean_grad2_ - tf.square(mean_grad1_) + ep)
                new_grad = tf.truediv(grad, rms)

            mean_grads1.append(mean_grad1)
            mean_grads2.append(mean_grad2)

            mean_grads1_updates.append(mean_grad1.assign(mean_grad1_))
            mean_grads2_updates.append(mean_grad2.assign(mean_grad2_))
            new_grads_and_vars.append((new_grad, var))
        train_op = self.optimizer.apply_gradients(new_grads_and_vars)
        updates = mean_grads1_updates + mean_grads2_updates + [train_op]
        return Operation(tf.group(*updates))
