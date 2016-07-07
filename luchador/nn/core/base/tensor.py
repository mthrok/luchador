from __future__ import absolute_import


class Tensor(object):
    """Provide shape information"""
    def __init__(self, tensor, shape, name=None):
        self.tensor = tensor
        self.shape = shape
        self.name = name

    def get_shape(self):
        return self.shape
