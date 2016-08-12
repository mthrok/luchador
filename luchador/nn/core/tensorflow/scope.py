from __future__ import absolute_import

import tensorflow as tf
from tensorflow import name_scope
from tensorflow import get_variable as _get_variable
from tensorflow import variable_scope
from tensorflow import get_variable_scope

from .tensor import Variable

__all__ = [
    'name_scope', 'get_variable', 'variable_scope', 'get_variable_scope']


def get_variable(name, shape=None, dtype=tf.float32,
                 initializer=None, regularizer=None, trainable=True, **kwargs):
    variable = _get_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, trainable=trainable, **kwargs)
    return Variable(variable, shape=shape, dtype=dtype)
