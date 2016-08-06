from __future__ import absolute_import


class Tensor(object):
    """Tensor class which provides shape information.

    This class was introduced to provide easy shape inference to Theano Tensors
    while having the same interface as Tensorflow Variables.
    """
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
