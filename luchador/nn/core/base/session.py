from __future__ import absolute_import

__all__ = ['Session']


class Session(object):
    """Defines common interface for computation handler

    Each backend must implement the following methods:
      - __init__: Construct instance and initialize session
      - run: Run computation
      - initialize: Initialize Variable-s
    """
    def __init__(self):
        raise NotImplementedError(
            '{}.{} is not implemented yet.'
            .format(type(self).__module__, type(self).__name__)
        )

    def run(self, name, inputs, outputs, updates, givens):
        raise NotImplementedError(
            '`run` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )

    def initialize(self):
        raise NotImplementedError(
            '`initialize` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )
