"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import abc
import logging

import luchador.util
from .scope import variable_scope
from .node import Node

__all__ = ['BaseLayer', 'fetch_layer']
_LG = logging.getLogger(__name__)


class BaseLayer(luchador.util.StoreMixin, Node):
    """Define common interface (``build``, ``parameters`` ...) of Layer"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(BaseLayer, self).__init__()
        self._store_args(**kwargs)

    def __call__(self, input_tensor):
        """Convenience method to call `build`"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build layer computation graph on top of the given tensor

        Parameters
        ----------
        input_tensor : Tensor
            Tensor object. Requirement for this object (such as shape and
            dtype) varies from Layer types.

        Returns
        -------
        Tensor
            Tensor which holds the output of built computation
        """
        _LG.info(
            '  Building %s(%s) on %s',
            type(self).__name__, self.args['scope'], input_tensor)
        self.input = input_tensor
        with variable_scope(self.args['scope']):
            self.output = self._build(input_tensor)
        _LG.info('    Built %s', self.output)
        return self.output

    @abc.abstractmethod
    def _build(self, input_tensor):
        """Actual build method"""
        raise NotImplementedError(
            '`_build` method is not implemented for {}'.format(self.__class__)
        )


def fetch_layer(name):
    """Get ``Layer`` class by name

    Parameters
    ----------
    name : str
        Type of ``Layer`` to get

    Returns
    -------
    type
        ``Layer`` type found

    Raises
    ------
    ValueError
        When ``Layer`` class with the given type is not found
    """
    for class_ in luchador.util.fetch_subclasses(BaseLayer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Layer: {}'.format(name))
