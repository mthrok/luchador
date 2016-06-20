from __future__ import absolute_import

import tensorflow as tf

from ..base.tensor import Tensor as BaseTensor


class Tensor(BaseTensor):
    def __init__(self, tensor, shape=None, name=None):
        super(Tensor, self).__init__(tensor=tensor, shape=shape, name=name)

    def get_shape(self):
        return self.tensor.get_shape().as_list()


class Input(Tensor):
    def __init__(self, dtype=tf.float32, shape=None, name=None):
        super(Input, self).__init__(tensor=None, shape=shape, name=name)

        self.dtype = dtype

    def __call__(self):
        return self.build()

    def build(self):
        if self.tensor is None:
            self.tensor = tf.placeholder(
                dtype=self.dtype, shape=self.shape, name=self.name)
        return self
