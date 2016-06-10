from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf
from tensorflow.contrib import layers

from .core import TFLayer
from .utils import get_function_args

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class Dense(TFLayer):
    def __init__(self, n_nodes, initializers=None):
        super(Dense, self).__init__(args=get_function_args())

    def _instantiate_parameter_variables(self, n_inputs):
        args = self.args
        b_shape = (args['n_nodes'],)
        w_shape = (n_inputs, args['n_nodes'])

        if args.get('initializers'):
            b_initializer = args['initializers']['bias']
            w_initializer = args['initializers']['weight']
        else:
            b_initializer = tf.constant_initializer(0.1)
            w_initializer = layers.xavier_initializer()
        b = tf.get_variable(
            name='bias', shape=b_shape, initializer=b_initializer)
        W = tf.get_variable(
            name='weight', shape=w_shape, initializer=w_initializer)
        self.parameter_variables = {'bias': b, 'weight': W}

    def __call__(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not self.parameter_variables:
            n_inputs = input_tensor.get_shape()[1].value
            self._instantiate_parameter_variables(n_inputs)

        prod = tf.matmul(input_tensor, self.parameter_variables['weight'])
        return tf.add(prod, self.parameter_variables['bias'], 'output')


class Conv2D(TFLayer):
    def __init__(self, filter_shape, n_filters, stride, padding='VALID',
                 initializers=None):
        """
        Args:
          filter_shape (tuple): [height, width]
          n_filters (int): #filters == #channels
          stride (int): stride
        """
        super(Conv2D, self).__init__(args=get_function_args())

    def _instantiate_parameter_variables(self, n_inputs):
        args = self.args
        b_shape = [args['n_filters']]
        w_shape = list(args['filter_shape']) + [n_inputs, args['n_filters']]
        if args.get('initializers'):
            b_initializer = args['initializers']['bias']
            w_initializer = args['initializers']['weight']
        else:
            b_initializer = tf.constant_initializer(0.1)
            w_initializer = layers.xavier_initializer_conv2d()
        b = tf.get_variable(
            name='bias', shape=b_shape, initializer=b_initializer)
        w = tf.get_variable(
            name='weight', shape=w_shape, initializer=w_initializer)
        self.parameter_variables = {'bias': b, 'weight': w}

    def __call__(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not self.parameter_variables:
            n_inputs = input_tensor.get_shape()[-1].value
            self._instantiate_parameter_variables(n_inputs)

        stride_shape = [1, self.args['stride'], self.args['stride'], 1]
        conv = tf.nn.conv2d(input_tensor, self.parameter_variables['weight'],
                            strides=stride_shape, padding=self.args['padding'])
        return tf.add(self.parameter_variables['bias'], conv, 'output')


class ReLU(TFLayer):
    def __init__(self):
        super(ReLU, self).__init__(args=get_function_args())

    def __call__(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        return tf.nn.relu(input_tensor, 'ouptut')


class Flatten(TFLayer):
    def __init__(self):
        super(Flatten, self).__init__(args=get_function_args())

    def __call__(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        in_shape = input_tensor.get_shape()
        n_nodes = reduce(lambda r, d: r*d.value, in_shape[1:], 1)
        out_shape = (-1, n_nodes)
        return tf.reshape(input_tensor, out_shape, 'output')


class TrueDiv(TFLayer):
    def __init__(self, denom):
        super(TrueDiv, self).__init__(args=get_function_args())

    def __call__(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        return tf.truediv(input_tensor, self.args['denom'], 'ouptut')
