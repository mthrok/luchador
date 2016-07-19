from __future__ import absolute_import

import tensorflow as tf
from tensorflow import (  # nopep8
    Session as _Session,
    get_default_session,
)

from ..base import Session as BaseSession


class Session(BaseSession):
    def __init__(self, graph=None, config=None, **kwargs):
        self.session = _Session('', graph, config)

    def run(self, outputs, inputs={}, updates={}, givens={}):
        if updates:
            outputs = outputs + updates
        if givens:
            inputs = inputs.update(givens)
        fetches = [output.tensor for output in outputs]
        feed_dict = {key.tensor: value for key, value in inputs.items()}
        return self.session.run(fetches, feed_dict=feed_dict)

    def close(self):
        return self.session.close()

    def initialize(self):
        self.session.run(tf.initialize_all_variables())
