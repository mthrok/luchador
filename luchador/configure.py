from __future__ import absolute_import


_NN_BACKEND = 'tensorflow'


def set_nn_backend(backend):
    if backend not in ['tensorflow', 'theano']:
        raise ValueError('Backend must be either "tensorflow" or "theano"')

    global _NN_BACKEND
    _NN_BACKEND = backend


def get_nn_backend():
    return _NN_BACKEND


def sane_gym_import():
    """Import Gym while avoiding polluting root logger"""
    import logging
    root_lg = logging.getLogger()
    root_handlers = root_lg.handlers
    root_lg.handlers = []

    import gym  # nopep8
    gym_lg = logging.getLogger('gym')
    gym_lg.handlers = root_lg.handlers
    gym_lg.propagate = False
    root_lg.handlers = root_handlers
