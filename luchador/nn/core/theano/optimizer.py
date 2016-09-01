from __future__ import absolute_import

from collections import OrderedDict

import theano
import theano.tensor as T

from ..base import (
    get_optimizer,
    Optimizer,
)
from .scope import get_variable, variable_scope
from .initializer import Constant
from .wrapper import Operation

__all__ = [
    'BaseOptimizer', 'get_optimizer',
    'SGD', 'RMSProp', 'GravesRMSProp', 'NeonRMSProp',
]


class BaseOptimizer(Optimizer):
    def minimize(self, loss, wrt, **kwargs):
        grads_and_vars = self.compute_gradients(loss, wrt, **kwargs)
        return self.apply_gradients(grads_and_vars)

    def compute_gradients(self, loss, wrt, **kwargs):
        # TODO: Add support for single wrt
        loss, wrt = loss.get(), [v.get() for v in wrt]
        grads = theano.grad(loss, wrt)
        return [(grad, var) for grad, var in zip(grads, wrt)]


class SGD(BaseOptimizer):
    def __init__(self, learning_rate, name='SGD', **kwargs):
        super(SGD, self).__init__(
            learning_rate=learning_rate, name=name)

    def apply_gradients(self, grads_and_vars):
        updates = OrderedDict()
        for grad, var in grads_and_vars:
            updates[var] = var - self.args['learning_rate'] * grad
        return Operation(op=updates)


class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay=0.95, momentum=None,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        super(RMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, momentum=momentum,
            epsilon=epsilon, name=name)

    def apply_gradients(self, grads_and_vars):
        # TODO: Save intermediate Variables in slot
        updates = OrderedDict()
        args = self.args
        d, mom = args['decay'], args['momentum']
        ep, lr = args['epsilon'], args['learning_rate']
        for grad, var in grads_and_vars:
            value = var.get_value(borrow=True)
            with variable_scope(args['name']):
                name = '{}_grad_squared_mean'.format(var.name)
                mean_grad2_ = get_variable(
                    name=name, shape=value.shape, dtype=value.dtype,
                    initializer=Constant(0), broadcastable=var.broadcastable)
                mean_grad2 = mean_grad2_.get()

                name = '{}_delta'.format(var.name)
                delta_ = get_variable(
                    name=name, shape=value.shape, dtype=value.dtype,
                    initializer=Constant(0), broadcastable=var.broadcastable)
                delta = delta_.get()

                new_mean_grad2 = d * mean_grad2 + (1.0 - d) * T.square(grad)

                rms = T.sqrt(new_mean_grad2 + ep)
                new_grad = grad / rms

                delta_var = -lr * new_grad
                new_delta = mom * delta + (1.0 - mom) * delta_var
                new_var = var + new_delta

            updates[mean_grad2] = new_mean_grad2
            updates[delta] = new_delta
            updates[var] = new_var
        return Operation(op=updates)


class NeonRMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay=0.95, epsilon=1e-6,
                 name='NeonRMSProp', **kwargs):
        super(NeonRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, epsilon=epsilon, name=name)

    def apply_gradients(self, grads_and_vars):
        # TODO: Save intermediate Variables in slot
        updates = OrderedDict()
        args = self.args
        d, ep, lr = args['decay'], args['epsilon'], args['learning_rate']
        for grad, var in grads_and_vars:
            value = var.get_value(borrow=True)
            with variable_scope(args['name']):
                name = '{}_grad_squared_mean'.format(var.name)
                mean_grad2_ = get_variable(
                    name=name, shape=value.shape, dtype=value.dtype,
                    initializer=Constant(0), broadcastable=var.broadcastable)
                mean_grad2 = mean_grad2_.get()

                new_mean_grad2 = d * mean_grad2 + (1.0 - d) * T.square(grad)

                rms = T.sqrt(new_mean_grad2 + ep) + ep
                new_grad = grad / rms

                delta = -lr * new_grad
                new_var = var + delta

            updates[mean_grad2] = new_mean_grad2
            updates[var] = new_var
        return Operation(op=updates)


class GravesRMSProp(BaseOptimizer):
    """RMSProp used in DQN paper[1] and described in A.Graves paper [2]

    [1] https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner/blob/4b9f5a79b03ea0cfc512ed1c11f1b00bc875bc57/dqn/NeuralQLearner.lua#L265  # nopep8
    [2] http://arxiv.org/pdf/1308.0850v5.pdf
    """
    def __init__(self, learning_rate,
                 decay1=0.95, decay2=0.95,
                 epsilon=1e-2, name='RMSProp'):
        # TODO: Add support for momentum
        super(GravesRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay1=decay1, decay2=decay2, epsilon=epsilon, name=name)
        self.decay1 = decay1
        self.decay2 = decay2
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def apply_gradients(self, grads_and_vars):
        # TODO: Save intermediate Variables in slot
        updates = OrderedDict()
        args = self.args
        d1, d2 = args['decay1'], args['decay2']
        ep, lr = args['epsilon'], args['learning_rate']
        for grad, var in grads_and_vars:
            value = var.get_value(borrow=True)
            with variable_scope(args['name']):
                name = '{}_grad_mean'.format(var.name)
                mean_grad1_ = get_variable(
                    name=name, shape=value.shape, dtype=value.dtype,
                    initializer=Constant(0), broadcastable=var.broadcastable)
                mean_grad1 = mean_grad1_.get()

                name = '{}_grad_squared_mean'.format(var.name)
                mean_grad2_ = get_variable(
                    name=name, shape=value.shape, dtype=value.dtype,
                    initializer=Constant(0), broadcastable=var.broadcastable)
                mean_grad2 = mean_grad2_.get()

                new_mean_grad1 = d1 * mean_grad1 + (1.0 - d1) * grad
                new_mean_grad2 = d2 * mean_grad2 + (1.0 - d2) * T.square(grad)

                rms = T.sqrt(new_mean_grad2 - T.square(new_mean_grad1) + ep)
                new_grad = grad / rms

                delta_var = -lr * new_grad
                new_var = var + delta_var

            updates[mean_grad1] = new_mean_grad1
            updates[mean_grad2] = new_mean_grad2
            updates[var] = new_var
        return Operation(op=updates)
