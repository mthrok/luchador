from __future__ import division
from __future__ import absolute_import

import logging

from .utils import get_function_args

_LG = logging.getLogger(__name__)

__all__ = ['Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv']


class CopyMixin(object):
    def store_args(self, args):
        """
        Args:
            args (dict): Arguments used to initialize the subclass instance.
            This will be used to recreate a subclass instance.
        """
        self.args = args

    def copy(self):
        """Create new layer instance with the same configuration"""
        return type(self)(**self.args)


class BaseLayer(CopyMixin, object):
    def __init__(self, args):
        super(BaseLayer, self).__init__()
        self.store_args(args)
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
    def __init__(self, filter_height, filter_width, n_filters, stride,
                 padding='VALID', initializers=None):
        """
        Args:
          filter_shape (tuple): [height, width]
          n_filters (int): #filters == #channels
          stride (int): stride
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
