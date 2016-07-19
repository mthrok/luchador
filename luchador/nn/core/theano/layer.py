from __future__ import division
from __future__ import absolute_import

import logging

import theano
import theano.tensor as T

from ..base import (
    ReLU as BaseReLU,
    Dense as BaseDense,
    Conv2D as BaseConv2D,
    Flatten as BaseFlatten,
    TrueDiv as BaseTrueDiv,
)
from . import scope as scp
from .tensor import Tensor
from .initializer import (
    Constant,
    Xavier,
    XavierConv2D,
)

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class Dense(BaseDense):
    def _instantiate_parameter_variables(self, n_inputs):
        args = self.args
        b_shape = (args['n_nodes'], )
        w_shape = (n_inputs, args['n_nodes'])

        given = args.get('initializers')
        b_init = given['bias'] if given else Constant(0.1)
        w_init = given['weight'] if given else Xavier()

        b = scp.get_variable(name='bias', shape=b_shape, initializer=b_init)
        W = scp.get_variable(name='weight', shape=w_shape, initializer=w_init)
        self.parameter_variables['weight'] = Tensor(tensor=W, shape=w_shape)
        self.parameter_variables['bias'] = Tensor(tensor=b, shape=b_shape)

    def build(self, input):
        """
        Args:
          input (ShapedTensor): Placeholder which holds the actual tensor
          Output: Output object
        """
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not len(input.shape) == 2:
            raise ValueError('Input tensor must be 2D. '
                             'Insted of {}'.format(len(input.shape)))

        if not self.parameter_variables:
            self._instantiate_parameter_variables(input.shape[1])

        prod = T.dot(input.tensor, self.parameter_variables['weight'].tensor)
        output_tensor = prod + self.parameter_variables['bias'].tensor
        output_shape = (input.shape[0], self.args['n_nodes'])
        return Tensor(output_tensor, shape=output_shape, name='output')


class Conv2D(BaseConv2D):
    ###########################################################################
    # Parameter validation
    def _validate_padding(self, padding):
        if isinstance(padding, int):
            return
        if padding in ['valid', 'full', 'half']:
            return
        try:
            p0, p1 = padding[0], padding[1]
            if (isinstance(p0, int) and isinstance(p1, int)):
                return
        except Exception:
            pass

        raise ValueError(
            '`padding` must be either str ("valid", "full" or "half"), '
            'int or tuple of two int')

    def _validate_strides(self, strides):
        if isinstance(strides, int):
            return
        try:
            p0, p1 = strides[0], strides[1]
            if (isinstance(p0, int) and isinstance(p1, int)):
                return
        except Exception:
            pass

        raise ValueError('`strides` must be either int or tuple of two int')

    def _validate_args(self, args):
        if isinstance(args.get('padding'), str):
            args['padding'] = args['padding'].lower()
            if args['padding'] == 'same':
                args['padding'] = 'half'
        self._validate_padding(args['padding'])
        self._validate_strides(args['strides'])

    ###########################################################################
    def _instantiate_parameter_variables(self, n_inputs):
        args = self.args
        b_shape = (args['n_filters'],)
        w_shape = (args['n_filters'], n_inputs,
                   args['filter_height'], args['filter_width'])
        given = args.get('initializers')
        b_init = given['bias'] if given else Constant(0.1)
        w_init = given['weight'] if given else XavierConv2D()

        b = scp.get_variable(name='bias', shape=b_shape, initializer=b_init)
        W = scp.get_variable(name='weight', shape=w_shape, initializer=w_init)
        self.parameter_variables['weight'] = Tensor(tensor=W, shape=w_shape)
        self.parameter_variables['bias'] = Tensor(tensor=b, shape=b_shape)

    def _get_subsample(self):
        if isinstance(self.args['strides'], int):
            return (self.args['strides'], self.args['strides'])
        return self.args['strides']

    def _get_border_mode(self):
        return self.args['padding']

    def _get_output_shape(self, input_shape, filter_shape):
        """Compute output shape

        Args:
          input_shape(tuple): (batch, n_input_channels, row, col)
          filter_shape(tuple): (n_filters, n_input_channels, rows, cols)
        """
        # TODO: Add warning if
        # parts of image are not covered because of subsampling
        f_row, f_col = filter_shape[2:4]
        in_row, in_col = input_shape[2:4]
        sub_row, sub_col = self._get_subsample()
        border_mode = self._get_border_mode()
        # Process padding
        if border_mode in ['full', 'valid']:
            pass
        elif border_mode == 'half':
            in_row += 2 * (f_row // 2)
            in_col += 2 * (f_col // 2)
        elif isinstance(border_mode, int):
            in_row += 2 * border_mode
            in_col += 2 * border_mode
        else:
            in_row += 2 * border_mode[0]
            in_col += 2 * border_mode[1]
        # Process convolution
        if border_mode == 'full':
            out_row = (in_row + f_row - 2) // sub_row + 1
            out_col = (in_col + f_col - 2) // sub_col + 1
        else:
            out_row = (in_row - f_row) // sub_row + 1
            out_col = (in_col - f_col) // sub_col + 1
        # Reconstruct
        n_batches, n_filters = input_shape[0], filter_shape[0]
        output_shape = (n_batches, n_filters, out_row, out_col)
        return output_shape

    def build(self, input):
        """
        Args:
          input (ShapedTensor): Placeholder which holds the actual tensor.
            Expected shape is (batch, stack, row, col).
          Output: Output object
        """
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        _LG.debug('    input_shape: {}'.format(input.shape))
        _LG.debug('    border_mode: {}'.format(self._get_border_mode()))
        if not len(input.shape) == 4:
            raise ValueError('Input tensor must be 4D. '
                             'Insted of {}'.format(len(input.shape)))

        if not self.parameter_variables:
            self._instantiate_parameter_variables(input.shape[1])

        filters = self.parameter_variables['weight'].tensor
        filter_shape = filters.get_value().shape
        subsample = self._get_subsample()
        border_mode = self._get_border_mode()

        conv = T.nnet.conv2d(
            input.tensor, filters=filters,
            input_shape=input.shape, filter_shape=filter_shape,
            border_mode=border_mode, subsample=subsample)

        bias = self.parameter_variables['bias'].tensor
        bias = bias.dimshuffle(('x', 0, 'x', 'x'))
        output_tensor = bias + conv
        output_shape = self._get_output_shape(input.shape, filter_shape)
        _LG.debug('    output_shape: {}'.format(output_shape))
        return Tensor(output_tensor, output_shape, name='output')


class ReLU(BaseReLU):
    def build(self, input):
        """
        Args:
          input (ShapedTensor): Placeholder object.
          Output: Output object
        """
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        output_tensor = T.nnet.relu(input.tensor)
        _LG.debug('    input_shape: {}'.format(input.shape))
        return Tensor(output_tensor, input.shape, name='output')


class Flatten(BaseFlatten):
    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        _LG.debug('    Input shape: {}'.format(input.shape))
        n_nodes = int(reduce(lambda r, d: r*d, input.shape[1:], 1))
        _LG.debug('    #Nodes     : {}'.format(n_nodes))
        output_shape = (input.shape[0] or -1, n_nodes)
        output_tensor = T.reshape(input.tensor, output_shape)
        _LG.debug('    output_shape: {}'.format(output_shape))
        return Tensor(output_tensor, output_shape, name='output')


class TrueDiv(BaseTrueDiv):
    def _instantiate_denominator(self):
        dtype = self.args['dtype'] or theano.config.floatX
        self.denom = T.constant(self.args['denom'], dtype=dtype, name='denominator')

    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if self.denom is None:
            self._instantiate_denominator()
        output_tensor = input.tensor / self.args['denom']
        return Tensor(output_tensor, input.shape, name='output')
