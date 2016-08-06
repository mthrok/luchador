from __future__ import absolute_import


class Tensor(object):
    """Provide shape information"""
    def __init__(self, tensor, shape, name, dtype):
        self.tensor = tensor
        self.shape = shape
        self.name = name
        self.dtype = dtype

    def get_shape(self):
        return self.shape


class Operation(object):
    def __init__(self, op):
        self.op = op
