from __future__ import absolute_import

from collections import Counter

import theano
import theano.tensor as T

from ..base.tensor import (
    Wrapper as BaseWrapper,
    Operation,
)

__all__ = ['Tensor', 'Input', 'Operation']

VARIABLE_COUNTER = Counter()


class Variable(BaseWrapper):
    """Wrapps for SharedVariable objects"""
    def __init__(self, variable):
        val = variable.get_value()
        name_ = variable.name
        count = VARIABLE_COUNTER[name_]
        name = '{}:{}'.format(name_, count)
        super(Variable, self).__init__(variable, val.shape, name, val.dtype)


class Tensor(BaseWrapper):
    """Wrapps Tensor objects"""
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
            self.set(T.scalar(name=self.name, dtype=self.dtype))
            return self

        dim = len(self.shape)
        if dim == 1:
            t = T.vector(name=self.name, dtype=self.dtype)
        elif dim == 2:
            t = T.matrix(name=self.name, dtype=self.dtype)
        elif dim == 3:
            t = T.tensor3(name=self.name, dtype=self.dtype)
        elif dim == 4:
            t = T.tensor4(name=self.name, dtype=self.dtype)
        else:
            raise ValueError('shape length must be smaller than 5')
        self.set(t)
        return self
