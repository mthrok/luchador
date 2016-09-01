from __future__ import absolute_import

import tensorflow as tf  # nopep8

from ..base import (
    get_optimizer,
    Optimizer,
)
from .wrapper import Operation

__all__ = [
    'BaseOptimizer', 'get_optimizer',
    'SGD', 'RMSProp', 'GravesRMSProp', 'NeonRMSProp'
]


class BaseOptimizer(Optimizer):
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
        loss = loss.get()
        # TODO: Add support for single tensor
        var_list = [v.get() for v in wrt] if wrt else None
        grads_and_vars = self.optimizer.compute_gradients(
            loss, var_list=var_list, **kwargs)
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, **kwargs):
        minimize_op = self.optimizer.apply_gradients(grads_and_vars, **kwargs)
        return Operation(minimize_op)


class SGD(BaseOptimizer):
    def __init__(self, learning_rate, name='SGD', **kwargs):
        super(SGD, self).__init__(learning_rate=learning_rate, name=name)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate, name=name, **kwargs)


class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate,
                 decay=0.95, momentum=None,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        super(RMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, momentum=momentum, epsilon=epsilon, name=name)
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=decay, momentum=momentum,
            epsilon=epsilon, **kwargs)


class NeonRMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay=0.95, epsilon=1e-6,
                 name='NeonRMSProp', **kwargs):
        super(NeonRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, epsilon=epsilon, name=name)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate, name=name)
        self.decay = decay
        self.epsilon = epsilon

    def apply_gradient(self, grads_and_vars, **kwargs):
        # TODO: Save intermediate Variables in slot
        mean_grads2, mean_grads2_updates = [], []
        new_grads_and_vars = [], []
        d2, ep = self.decay2, self.epsilon
        for grad, var in grads_and_vars:
            with tf.variable_scope(self.name):
                name = '{}_squared_mean'.format(grad.name)
                mean_grad2 = tf.get_variable(
                    name=name, shape=grad.shape, dtype=grad.dtype,
                    initializer=tf.constatnt_initializer(0))

                new_mean_grad2 = d2 * mean_grad2 + (1.0 - d2) * tf.square(grad)

                rms = tf.sqrt(new_mean_grad2 + ep) + ep
                new_grad = tf.truediv(grad, rms)

            mean_grads2.append(mean_grad2)

            mean_grads2_updates.append(mean_grad2.assign(new_mean_grad2))
            new_grads_and_vars.append((new_grad, var))
        train_op = self.optimizer.apply_gradients(new_grads_and_vars)
        updates = mean_grads2_updates + [train_op]
        return Operation(tf.group(*updates))


class GravesRMSProp(BaseOptimizer):
    def __init__(self, learning_rate,
                 decay1=0.0, decay2=0.95, epsilon=1e-2,
                 name='GravesRMSProp', **kwargs):
        # TODO: Add support for momentum
        super(GravesRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay1=decay1, decay2=decay2, epsilon=epsilon, name=name)
        self.decay1 = decay1
        self.decay2 = decay2
        self.epsilon = epsilon
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate, name=name)

    def apply_gradient(self, grads_and_vars, **kwargs):
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

                new_mean_grad1 = d1 * mean_grad1 + (1.0 - d1) * grad
                new_mean_grad2 = d2 * mean_grad2 + (1.0 - d2) * tf.square(grad)

                rms = tf.sqrt(new_mean_grad2 - tf.square(new_mean_grad1) + ep)
                new_grad = tf.truediv(grad, rms)

            mean_grads1.append(mean_grad1)
            mean_grads2.append(mean_grad2)

            mean_grads1_updates.append(mean_grad1.assign(new_mean_grad1))
            mean_grads2_updates.append(mean_grad2.assign(new_mean_grad2))
            new_grads_and_vars.append((new_grad, var))

        train_op = self.optimizer.apply_gradients(new_grads_and_vars)
        updates = mean_grads1_updates + mean_grads2_updates + [train_op]
        return Operation(tf.group(*updates))
