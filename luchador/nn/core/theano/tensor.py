from __future__ import absolute_import

import theano
import theano.tensor as T

from ..base.tensor import Tensor as BaseTensor

__all__ = ['Tensor', 'Input']


class Tensor(BaseTensor):
    def __init__(self, tensor, shape=None, name=None, dtype=None):
        if tensor is not None:
            name = name or tensor.name
            dtype = dtype or tensor.dtype
        super(Tensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)


class Input(Tensor):
    def __init__(self, shape, name=None, dtype=theano.config.floatX):
        super(Input, self).__init__(
            tensor=None, shape=shape, name=name, dtype=dtype)

    def __call__(self):
        return self.build()

    def build(self):
        if not self.shape:
            self.tensor = T.scalar(name=self.name, dtype=self.dtype)
            return self

        dim = len(self.shape)
        if dim == 1:
            self.tensor = T.vector(name=self.name, dtype=self.dtype)
        elif dim == 2:
            self.tensor = T.matrix(name=self.name, dtype=self.dtype)
        elif dim == 3:
            self.tensor = T.tensor3(name=self.name, dtype=self.dtype)
        elif dim == 4:
            self.tensor = T.tensor4(name=self.name, dtype=self.dtype)
        else:
            raise ValueError('shape length must be smaller than 5')
        return self


class Operation(object):
    def __init__(self, op):
        self.op = op
