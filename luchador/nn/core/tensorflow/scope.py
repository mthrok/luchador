from __future__ import absolute_import

import tensorflow as tf
from tensorflow import (
    name_scope,
    get_variable as _get_variable,
    VariableScope,
    variable_scope,
    get_variable_scope,
)

from .wrapper import Variable
from .initializer import TFInitializer

__all__ = ['name_scope', 'get_variable', 'variable_scope',
           'VariableScope', 'get_variable_scope']


def get_variable(name, shape=None, dtype=tf.float32,
                 initializer=None, regularizer=None, trainable=True, **kwargs):
    if isinstance(initializer, TFInitializer):
        initializer = initializer.get()

    variable = _get_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, trainable=trainable, **kwargs)

    dtype = variable.dtype.as_numpy_dtype()
    return Variable(variable, shape=shape, dtype=dtype)
