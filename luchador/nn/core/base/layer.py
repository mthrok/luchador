from __future__ import division
from __future__ import absolute_import

import logging

from .utils import get_function_args

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class CopyMixin(object):
    def _store_args(self, args):
        """
        Args:
            args (dict): Arguments used to initialize the subclass instance.
            This will be used to recreate a subclass instance.
        """
        self._validate_args(args)
        self.args = args

    def _validate_args(self, args):
        """Perform parameter validation"""
        pass

    def copy(self):
        """Create new layer instance with the same configuration"""
        return type(self)(**self.args)


class BaseLayer(CopyMixin, object):
    def __init__(self, args):
        super(BaseLayer, self).__init__()
        self._store_args(args)
        self.parameter_variables = {}

    def __call__(self, input_tensor):
        """Build layer computation graph on top of the given tensor"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build layer computation graph on top of the given tensor"""
        raise NotImplementedError(
            '`build` method is not implemented for {} class.'
            .format(type(self).__name__)
        )


###############################################################################
class Dense(BaseLayer):
    def __init__(self, n_nodes, initializers=None):
        super(Dense, self).__init__(args=get_function_args())


class Conv2D(BaseLayer):
    def __init__(self, filter_height, filter_width, n_filters, strides,
                 padding='VALID', initializers=None, **kwargs):
        """
        Args:
          filter_height (int): filter height
          filter_weight (int): filter weight
          n_filters (int): #filters == #output channels
          strides (int, tuple of two int, or tuple of four int): stride
            - When given type is int, the out put is subsampled by this factor
              in both width and height direction.
            - When given type is tuple of two int, subsapmled `strides[0]` in
              height direction and `striders[1]` in width direction.
            - [Tensorflow only] When given type is tuple of four int, it must
              be consistent with the input data format. That is:
              - data_format=='NHWC' (default): [batch, height, width, channel]
              - data_format=='NCHW': [batch, channel, height, width]
          padding:
            - [tensorflow] (str): Either 'SAME' or 'VALID'
            - [theano] (str or int or tuple of two int): See Theano doc
          kwargs: other optional arguments passed to conv2d functions.
            - Tensorflow: 'use_cudnn_on_gpu', 'data_format', 'name'
        """
        super(Conv2D, self).__init__(args=get_function_args())


class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__(args=get_function_args())


class Flatten(BaseLayer):
    def __init__(self):
        super(Flatten, self).__init__(args=get_function_args())


class TrueDiv(BaseLayer):
    def __init__(self, denom):
        super(TrueDiv, self).__init__(args=get_function_args())
