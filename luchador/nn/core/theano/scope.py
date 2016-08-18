from __future__ import absolute_import

import logging

import theano
from theano import config

from .initializer import Normal
from .wrapper import Variable

_LG = logging.getLogger(__name__)

_VARIABLES = {}
_CURRENT_REUSE_FLAG = False
_CURRENT_VARIABLE_SCOPE = ''


def _set_flag(flag):
    global _CURRENT_REUSE_FLAG
    _CURRENT_REUSE_FLAG = flag


def _set_scope(scope):
    global _CURRENT_VARIABLE_SCOPE
    _CURRENT_VARIABLE_SCOPE = scope


def _get_flag():
    return _CURRENT_REUSE_FLAG


def _get_scope():
    return _CURRENT_VARIABLE_SCOPE


def _reset():
    """Reset variable scope. For testing"""
    _set_flag(False)
    _set_scope('')
    global _VARIABLES
    _VARIABLES = {}


class NameScope(object):
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


class VariableScope(object):
    def __init__(self, reuse, name=''):
        self.name = name
        self.reuse = reuse

        self.previous_scopes = []
        self.previous_reuse_flags = []

    def reuse_variables(self):
        _set_flag(True)

    def _open(self):
        self.previous_scopes.append(_get_scope())
        self.previous_reuse_flags.append(_get_flag())
        _set_scope(self.name)
        _set_flag(self.reuse)

    def _close(self):
        _set_scope(self.previous_scopes.pop())
        _set_flag(self.previous_reuse_flags.pop())

    def __enter__(self):
        self._open()
        _LG.debug('Current Scope: {}'.format(_CURRENT_VARIABLE_SCOPE))
        return self

    def __exit__(self, type, value, traceback):
        self._close()
        _LG.debug('Current Scope: {}'.format(_CURRENT_VARIABLE_SCOPE))


def variable_scope(name_or_scope, reuse=None):
    if isinstance(name_or_scope, VariableScope):
        return name_or_scope

    scope = (
        '{}/{}'.format(_get_scope(), name_or_scope)
        if _get_scope() else name_or_scope
    )
    return VariableScope(reuse, scope)


def name_scope(name):
    return NameScope()


def get_variable_scope():
    """Return the current variable scope"""
    return VariableScope(_get_flag(), _get_scope())


def get_variable(name, shape=None, dtype=config.floatX,
                 initializer=None, regularizer=None, trainable=None, **kwargs):
    if trainable:
        raise NotImplementedError('trainable option is not yet implemented.')
    if regularizer:
        raise NotImplementedError('regularizer option is not yet implemented.')

    # 1. Check the current variable scope
    scope = _get_scope()
    name = '{}/{}'.format(scope, name) if scope else name
    if _get_flag():  # Search for an existing variable
        if name not in _VARIABLES:
            raise ValueError(
                'Variable {} does not exist, disallowed. '
                'Did you mean to set reuse=None in VarScope?'
                .format(name)
            )
    else:  # Create new variable
        if name in _VARIABLES:
            raise ValueError(
                'Variable {} already exists, disallowed. '
                'Did you mean to set reuse=True in VarScope?'
                .format(name)
            )
        if shape is None:
            raise ValueError(
                'Shape of a new variable ({}) must be fully defined, '
                'but instead was {}.'.format(name, shape))
        if not initializer:
            initializer = Normal(dtype=dtype)

        _VARIABLES[name] = Variable(
            theano.shared(
                value=initializer.sample(shape),
                name=name, allow_downcast=True, **kwargs)
        )

    return _VARIABLES[name]
