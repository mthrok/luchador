from __future__ import absolute_import

import os

_NN_DTYPE = os.environ.get('LUCHADOR_NN_DTYPE', 'float32')
_NN_BACKEND = os.environ.get('LUCHADOR_NN_BACKEND', 'tensorflow')
_NN_CONV_FORMAT = os.environ.get('LUCHADOR_NN_CONV_FORMAT', 'NCHW')

__all__ = [
    'get_nn_backend', 'get_nn_conv_format', 'get_nn_dtype',
 ]


def get_nn_backend():
    return _NN_BACKEND


def get_nn_conv_format():
    return _NN_CONV_FORMAT


def get_nn_dtype():
    return _NN_DTYPE
