from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf
from tensorflow.contrib import layers

from ..base import (
    ReLU as BaseReLU,
    Dense as BaseDense,
    Conv2D as BaseConv2D,
    Flatten as BaseFlatten,
    TrueDiv as BaseTrueDiv,
)
from .tensor import Tensor

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class Dense(BaseDense):
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

    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not self.parameter_variables:
            n_inputs = input.get_shape()[1]
            self._instantiate_parameter_variables(n_inputs)

        prod = tf.matmul(input.tensor, self.parameter_variables['weight'])
        output_tensor = tf.add(prod, self.parameter_variables['bias'], 'output')
        return Tensor(tensor=output_tensor)


class Conv2D(BaseConv2D):
    def _validate_padding(self, padding):
        if padding not in ['SAME', 'VALID']:
            raise ValueError('`padding` must be either "SAME" or "VALID"')

    def _validate_strides(self, strides):
        if isinstance(strides, int):
            return
        try:
            if (
                    len(strides) in [2, 4] and
                    all(map(lambda s: issubclass(s, int), strides))
            ):
                return
        except Exception:
            pass
        raise ValueError(
            '`strides` must be either int, '
            'tuple of two ints or tuple of four ints'
        )

    def _validate_data_format(self, data_format):
        if data_format not in ['NHWC', 'NCHW']:
            raise ValueError(
                '`data_format` mut be either "NCHW" or "NHWC".')

    def _validate_args(self, args):
        args['padding'] = args['padding'].upper()
        self._validate_padding(args['padding'])
        self._validate_strides(args['strides'])
        self._validate_data_format(args.get('data_format', 'NHWC'))

    ###########################################################################
    def _get_strides(self):
        strides = self.args['strides']
        data_format = self.args.get('data_format', 'NHWC')

        if data_format == 'NHWC':
            if isinstance(strides, int):
                return [1, strides, strides, 1]
            if len(strides) == 2:
                return [1, strides[0], strides[1], 1]
            return strides
        if isinstance(strides, int):
            return [1, 1, strides, strides]
        if len(strides) == 2:
            return [1, 1, strides[0], strides[1]]
        return strides

    def _instantiate_parameter_variables(self, n_inputs):
        args = self.args
        b_shape = [args['n_filters']]
        w_shape = (args['filter_height'], args['filter_width'],
                   n_inputs, args['n_filters'])

        given = args.get('initializers')
        b0 = given['bias'] if given else tf.constant_initializer(0.1)
        w0 = given['weight'] if given else layers.xavier_initializer_conv2d()

        # TODO: Add warning if
        # parts of image are not covered because of stride
        b = tf.get_variable(name='bias', shape=b_shape, initializer=b0)
        w = tf.get_variable(name='weight', shape=w_shape, initializer=w0)
        self.parameter_variables = {'bias': b, 'weight': w}

    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not self.parameter_variables:
            n_inputs = input.get_shape()[-1]
            self._instantiate_parameter_variables(n_inputs)

        strides = self._get_strides()
        name = self.args.get('name')
        cudnn = self.args.get('use_cudnn_on_gpu', True)
        data_format = self.args.get('use_cudnn_on_gpu', 'NHWC')
        conv = tf.nn.conv2d(
            input.tensor, self.parameter_variables['weight'],
            strides=strides, padding=self.args['padding'],
            use_cudnn_on_gpu=cudnn, data_format=data_format, name=name
        )
        output_t = tf.add(self.parameter_variables['bias'], conv, 'output')
        return Tensor(output_t)


class ReLU(BaseReLU):
    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        output_tensor = tf.nn.relu(input.tensor, 'ouptut')
        return Tensor(output_tensor)


class Flatten(BaseFlatten):
    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        in_shape = input.get_shape()
        n_nodes = reduce(lambda prod, dim: prod*dim, in_shape[1:], 1)
        out_shape = (-1, n_nodes)
        output_tensor = tf.reshape(input.tensor, out_shape, 'output')
        return Tensor(output_tensor)


class TrueDiv(BaseTrueDiv):
    def __init__(self, denom, dtype):
        super(TrueDiv, self).__init__(denom=denom)
        self.args['dtype'] = dtype
        self.denom = None

    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        print input
        print input.tensor
        if self.denom is None:
            self.denom = tf.constant(
                self.args['denom'], dtype=self.args['dtype'], name='denominator')
        output_tensor = tf.truediv(input.tensor, self.denom, 'ouptut')
        return Tensor(output_tensor)