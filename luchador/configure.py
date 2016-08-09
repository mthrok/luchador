from __future__ import absolute_import


_NN_BACKEND = 'tensorflow'


def set_nn_backend(backend):
    if backend not in ['tensorflow', 'theano']:
        raise ValueError('Backend must be either "tensorflow" or "theano"')

    global _NN_BACKEND
    _NN_BACKEND = backend


def get_nn_backend():
    return _NN_BACKEND
