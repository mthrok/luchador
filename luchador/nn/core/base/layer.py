from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

from .utils import get_function_args

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class CopyMixin(object):
    """Provide copy method which creates a copy of initialed instance"""
    def _store_args(self, args):
        """
        Args:
            args (dict): Arguments passed to __init__ of the subclass instance.
            This will be used to recreate a subclass instance.
        """
        self._validate_args(args)
        self.args = args

    def _validate_args(self, args):
        """Validate arguments"""
        pass

    def copy(self):
        """Create and initialize new instance with the same argument"""
        return type(self)(**self.args)


class BaseLayer(CopyMixin, object):
    """Defines common interface (copy and build methods) for Layer classes"""
    def __init__(self, args):
        """Validate and store arguments passed to subclass __init__ method"""
        super(BaseLayer, self).__init__()
        self._store_args(args)
        self.parameter_variables = OrderedDict()

    def __call__(self, input_tensor):
        """Build layer computation graph on top of the given tensor"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build layer computation graph on top of the given tensor"""
        raise NotImplementedError(
            '`build` method is not implemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )


###############################################################################
class Dense(BaseLayer):
    def __init__(self, n_nodes, initializers=None):
        """Initialize dense (also called affine, linear or inner product) layer.
        Activation functio such as ReLU is not included.

        Args:
          n_nodes (int): The number of internal neurons.
        """
        super(Dense, self).__init__(args=get_function_args())


class Conv2D(BaseLayer):
    def __init__(self, filter_height, filter_width, n_filters, strides,
                 padding='VALID', initializers=None, **kwargs):
        """Initialize 2D convolution layer.
        Args:
          filter_height (int): filter height (== row)
          filter_weight (int): filter weight (== column)
          n_filters (int): #filters (== #output channels)
          strides (int, tuple of two int, or tuple of four int): stride
            - When given type is int, the output is subsampled by this factor
              in both width and height direction.
            - When given type is tuple of two int, the output is subsapmled
              `strides[0]` in height direction and `striders[1]` in width
              direction.
            - [Tensorflow only] When given type is tuple of four int, it must
              be consistent with the input data format. That is:
              - data_format=='NHWC' (default): [batch, height, width, channel]
              - data_format=='NCHW': [batch, channel, height, width]
          padding:
            - [tensorflow] (str): Either 'SAME' or 'VALID'
            - [theano] (str or int or tuple of two int): See Theano doc
          kwargs:
            - Tensorflow: Arguments passed to tf.nn.conv2d.
              'use_cudnn_on_gpu' and 'name'
        """
        super(Conv2D, self).__init__(args=get_function_args())


class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__(args=get_function_args())


class Flatten(BaseLayer):
    def __init__(self):
        super(Flatten, self).__init__(args=get_function_args())


class TrueDiv(BaseLayer):
    def __init__(self, denom, dtype=None):
        super(TrueDiv, self).__init__(args=get_function_args())
        self.denom = None
