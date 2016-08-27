from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

from .common import CopyMixin

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class Layer(CopyMixin, object):
    """Defines common interface (copy and build) for Layer classes"""
    def __init__(self, **args):
        """Validate and store arguments passed to subclass __init__ method

        As these **args are used to create a copy of instance, arguments which
        cannot be passed to constructor should not be passed. In other way, the
        signature of this constructor must match to the signature of the
        constractor of subclass object being created.
        """
        super(Layer, self).__init__()
        self._store_args(args)

        self.initializers = OrderedDict()
        self.parameter_variables = OrderedDict()

    ###########################################################################
    # Functions for building computation graph
    def __call__(self, input_tensor):
        """Build layer computation graph on top of the given tensor"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build layer computation graph on top of the given tensor"""
        raise NotImplementedError(
            '`build` method is not implemented for {}'.format(self.__class__)
        )

    def __repr__(self):
        return "{{'name': '{}', 'args': {}}}".format(
            self.__class__.__name__, self.args)


###############################################################################
class Dense(Layer):
    def __init__(self, n_nodes, initializers={}):
        """Initialize dense layer.
        Activation function, such as ReLU is not included.
        Also called fully connected, affine, linear or inner product.

        Args:
          n_nodes (int): The number of internal neurons.
        """
        super(Dense, self).__init__(n_nodes=n_nodes, initializers=initializers)


class Conv2D(Layer):
    """Apply convolution to input"""
    def __init__(self, filter_height, filter_width, n_filters, strides,
                 padding='VALID', initializers={}, **kwargs):
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
        super(Conv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers, **kwargs)


class ReLU(Layer):
    """Applies Rectified Linear Unit"""
    def __init__(self):
        super(ReLU, self).__init__()


class Flatten(Layer):
    """Reshape batch into 2D (batch_size, n_features)"""
    def __init__(self):
        super(Flatten, self).__init__()


class TrueDiv(Layer):
    """Applies element wise division"""
    def __init__(self, denom, dtype=None):
        super(TrueDiv, self).__init__(denom=denom, dtype=None)
        self.denom = None  # denominator in Variable expression
