from __future__ import absolute_import

__all__ = ['Session']


class Session(object):
    def __init__(self):
        raise NotImplementedError(
            '{}.{} is not implemented yet.'
            .format(type(self).__module__, type(self).__name__)
        )

    def run(self, inputs, outputs, updates, givens):
        raise NotImplementedError(
            '`run` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )

    def initialize(self):
        pass
