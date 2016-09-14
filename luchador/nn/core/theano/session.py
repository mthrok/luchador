from __future__ import absolute_import

import warnings
from collections import OrderedDict

import theano

from ..base import Session as BaseSession
from .wrapper import (
    Tensor,
    Operation,
)

__all__ = ['Session']


def _get_full_class(cls):
    return '{}.{}'.format(cls.__module__, cls.__name__)

_TENSOR_CLASS_STR = _get_full_class(Tensor)
_OP_CLASS_STR = _get_full_class(Operation)


def _is_iteratable(l):
    try:
        list(l)
        return True
    except Exception:
        return False


def _parse_inputs(inputs):
    inputs_ = []
    if inputs is None:
        return inputs_

    if not _is_iteratable(inputs):
        inputs = [inputs]

    if isinstance(inputs, dict):
        for key, value in inputs.items():
            inputs_.append(key.get())
    elif isinstance(inputs, list):
        for key, value in inputs:
            inputs_.append(key.get())
    else:
        raise ValueError(
            '`inputs` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(inputs)))
    return inputs_


def _parse_outputs(outputs):
    if outputs is None:
        return []
    if not _is_iteratable(outputs):
        outputs = [outputs]
    return [o.get() for o in outputs]


def _parse_updates(updates):
    ret = OrderedDict()
    if updates is None:
        return ret

    if not _is_iteratable(updates):
        updates = [updates]

    for update in updates:
        if not isinstance(update, Operation):
            raise ValueError(
                '`updates` must be [list of] {}. Given: {}'
                .format(_OP_CLASS_STR, _get_full_class(type(update))))
        for shared_variable, new_expression in update.get().items():
            ret[shared_variable] = new_expression
    return ret


def _construct_function(inputs, outputs, updates, givens):
    inputs_ = _parse_inputs(inputs)
    outputs_ = _parse_outputs(outputs)
    updates_ = _parse_updates(updates)
    return theano.function(inputs_, outputs_, updates=updates_, givens=givens)


class Session(BaseSession):
    def __init__(self, **kwargs):
        self.functions = {}

    @property
    def graph(self):
        return None

    def run(self, outputs=[], inputs={}, updates=None, givens=None, name=None):
        if name in self.functions:
            function = self.functions[name]
        else:
            function = _construct_function(inputs, outputs, updates, givens)

        if name and name not in self.functions:
            self.functions[name] = function

        values = function(*inputs.values())
        if _is_iteratable(outputs):
            return values
        return values[0]

    def initialize(self):
        pass

    def close(self):
        warnings.warn('`close` does nothing in Theano backend.')
        pass
