from __future__ import division

import logging

import tensorflow as tf
from tensorflow.contrib import layers

from .core import TFLayer

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class Dense(TFLayer):
    def __init__(self, n_nodes, scope=None):
        args = {'n_nodes': n_nodes, 'scope': scope}
        super(Dense, self).__init__(name='dense', scope=scope, args=args)

    def build(self, input_tensor):
        _LG.debug('    Building {}/{}'.format(self.scope, self.name))
        n_input = input_tensor.get_shape()[1].value
        bias_shape = (self.args['n_nodes'], )
        weight_shape = (n_input, self.args['n_nodes'])

        with tf.variable_scope(self.get_scope()):
            b = tf.get_variable(
                name='bias',
                initializer=tf.constant(0.1, shape=bias_shape)
            )
            W = tf.get_variable(
                name='weight', shape=weight_shape,
                initializer=layers.xavier_initializer()
            )
            output_tensor = tf.add(b, tf.matmul(input_tensor, W), 'output')

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.parameter_variables = [b, W]
        return output_tensor


class Conv2D(TFLayer):
    def __init__(self, filter_shape, n_filters, stride,
                 padding='SAME', scope=None):
        """
        Args:
          filter_shape (tuple): [height, width]
          n_filters (int): #filters == #channels
          stride (int): stride
        """
        args = {
            'filter_shape': filter_shape,
            'n_filters': n_filters,
            'stride': stride,
            'padding': padding,
            'scope': scope,
        }
        super(Conv2D, self).__init__(name='conv2d', scope=scope, args=args)

    def build(self, input_tensor):
        _LG.debug('    Building {}/{}'.format(self.scope, self.name))
        args = self.args
        n_input = input_tensor.get_shape()[-1].value
        b_shape = [args['n_filters']]
        w_shape = list(args['filter_shape']) + [n_input, args['n_filters']]
        stride_shape = [1, args['stride'], args['stride'], 1]
        with tf.variable_scope(self.get_scope()):
            b = tf.get_variable(
                name='bias',
                initializer=tf.constant(0.1, shape=b_shape)
            )
            W = tf.get_variable(
                name='weight', shape=w_shape,
                initializer=layers.xavier_initializer_conv2d()
            )
            conv = tf.nn.conv2d(
                input_tensor, W, strides=stride_shape, padding=args['padding'])
            output_tensor = tf.add(b, conv, 'output')

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.parameter_variables = [b, W]
        return output_tensor


class ReLU(TFLayer):
    def __init__(self, scope=None):
        args = {'scope': scope}
        super(ReLU, self).__init__(name='relu', scope=scope, args=args)

    def build(self, input_tensor):
        _LG.debug('    Building {}/{}'.format(self.scope, self.name))
        with tf.variable_scope(self.get_scope()):
            output_tensor = tf.nn.relu(input_tensor, 'ouptut')

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        return output_tensor


class Flatten(TFLayer):
    def __init__(self, scope=None):
        args = {'scope': scope}
        super(Flatten, self).__init__(name='flatten', scope=scope, args=args)

    def build(self, input_tensor):
        _LG.debug('    Building {}/{}'.format(self.scope, self.name))
        input_shape = input_tensor.get_shape()
        n_nodes = reduce(lambda r, d: r*d.value, input_shape[1:], 1)
        output_shape = (-1, n_nodes)

        with tf.variable_scope(self.get_scope()):
            output_tensor = tf.reshape(input_tensor, output_shape, 'output')

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        return output_tensor


class TrueDiv(TFLayer):
    def __init__(self, denom, scope=None):
        args = {'denom': denom, 'scope': scope}
        super(TrueDiv, self).__init__(name='truediv', scope=scope, args=args)

    def build(self, input_tensor):
        _LG.debug('    Building {}/{}'.format(self.scope, self.name))
        denom = self.args['denom']
        with tf.variable_scope(self.get_scope()):
            output_tensor = tf.truediv(input_tensor, denom, 'ouptut')

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        return output_tensor
