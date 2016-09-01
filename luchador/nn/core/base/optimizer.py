from __future__ import absolute_import

from .common import CopyMixin
from luchador.common import get_subclasses

__all__ = ['Optimizer', 'get_optimizer']


class Optimizer(CopyMixin):
    """Defines common interface for gradient computation and application"""
    def __init__(self, **args):
        super(Optimizer, self).__init__()
        self._store_args(args)

    def minimize(self, loss, wrt, **kwargs):
        """Minimize loss with the given variables

        Args:
          loss (TensorWrapper):Loss value to be minimized
          wrt (list of TensorWrappers): Variables with which loss is minimzied

        Returns:
          OperationWrapper: Minimization operation
        """
        raise NotImplementedError(
            '`minimize` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def compute_gradients(self, loss, wrt, **kwargs):
        """Compute gradient of loss with respect to wrt.

        This is similar to Tensorflow's Optimizer.compute_gradient method.

        Args:
          loss (TensorWrapper): Loss value to compute gradients
          wrt (list of TensorWrapper):
            Variables with which gradients are computed

        Returns:
          List of Tensor pairs: Gradient and corresponding variable pairs.
            Each tensor is underlying object and not wrapped with
            TensorWrapper class.
        """
        raise NotImplementedError(
            '`compute_gradients` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply gradients to variables

        Args:
          grads_and_vars (list of Tensor pairs):
            Return value of compute_gradient method.

        Returns:
          OperationWrapper: Operation which updates variables.
        """
        raise NotImplementedError(
            '`apply_gradients` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )


def get_optimizer(name):
    for Class in get_subclasses(Optimizer):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Optimizer: {}'.format(name))
