from __future__ import absolute_import

import tensorflow as tf

from luchador import get_nn_dtype
from ..base.tensor import (
    Tensor as BaseTensor,
    Operation,
)

__all__ = ['Tensor', 'Input', 'Operation']


class Tensor(BaseTensor):
    def __init__(self, tensor, shape=None, name=None, dtype=None):
        if tensor is not None:
            name = name or tensor.name
            shape = shape or tensor.get_shape().as_list()
            dtype = dtype or tensor.dtype
        super(Tensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)


class Input(Tensor):
    def __init__(self, shape, name=None, dtype=None):
        dtype = dtype or get_nn_dtype()
        super(Input, self).__init__(
            tensor=None, shape=shape, name=name, dtype=dtype)

    def __call__(self):
        return self.build()

    def build(self):
        if self.tensor is None:
            self.tensor = tf.placeholder(
                dtype=self.dtype, shape=self.shape, name=self.name)
        return self
