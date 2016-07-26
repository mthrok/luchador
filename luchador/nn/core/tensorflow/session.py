from __future__ import absolute_import

import tensorflow as tf
from tensorflow import (  # nopep8
    Session as _Session,
    get_default_session,
)

from ..base import Session as BaseSession
from .tensor import Operation

__all__ = ['Session']


def _merge_outputs_and_updates(outputs, updates):
    if isinstance(updates, Operation):
        return outputs + [updates]
    if isinstance(updates, list):
        return outputs + updates
    _op = type(Operation)
    _op_name = '{}.{}'.format(_op.__module__, _op.__name__)
    _up = type(updates)
    _up_name = '{}.{}'.format(_up.__module__, _up.__name__)
    raise ValueError(
        'Unexpected `updates` type was given. '
        '`updates` must be [list of] {}. Found: {}.'
        .format(_op_name, _up_name)
    )


class Session(BaseSession):
    def __init__(self, graph=None, config=None, **kwargs):
        self.session = _Session('', graph, config)

    @property
    def graph(self):
        return self.session.graph

    def run(self, name, outputs=[], inputs={}, updates=None, givens=None):
        """

        Args:
          name (str): Not used. Compatibility for theano backend
          outputs (list of Tensors):
          inputs (dict):
          updates (Operation or list of Operations)
          givens (dict):
        """
        if givens:
            inputs.update(givens)
        if updates:
            outputs = _merge_outputs_and_updates(outputs, updates)
        fetches = [output.tensor for output in outputs]
        feed_dict = {key.tensor: value for key, value in inputs.items()}
        return self.session.run(fetches, feed_dict=feed_dict)

    def close(self):
        return self.session.close()

    def initialize(self):
        self.session.run(tf.initialize_all_variables())
