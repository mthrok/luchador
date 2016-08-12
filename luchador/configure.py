from __future__ import absolute_import

import os
import warnings

_NN_DTYPE = os.environ.get('LUCHADOR_NN_DTYPE', 'float32')
_NN_BACKEND = os.environ.get('LUCHADOR_NN_BACKEND', 'tensorflow')
_NN_CONV_FORMAT = os.environ.get('LUCHADOR_NN_CONV_FORMAT', 'NCHW')

__all__ = [
    'set_nn_backend', 'set_nn_conv_format', 'set_nn_dtype',
    'get_nn_backend', 'get_nn_conv_format', 'get_nn_dtype',
 ]


def set_nn_backend(backend):
    if backend not in ['tensorflow', 'theano']:
        raise ValueError('Backend must be either "tensorflow" or "theano"')

    global _NN_BACKEND
    _NN_BACKEND = backend


def get_nn_backend():
    return _NN_BACKEND


def set_nn_conv_format(conv_format):
    """Set default convolution data format for Tensorflow backend"""
    if _NN_BACKEND == 'theano' and not conv_format == 'NCHW':
        warnings.warn('conv format is only effective in Tensorflow backend.')

    if conv_format not in ['NCHW', 'NHWC']:
        raise ValueError('cnn_format must be either `NCHW` or `NHWC`')

    global _NN_CONV_FORMAT
    _NN_CONV_FORMAT = conv_format


def get_nn_conv_format():
    return _NN_CONV_FORMAT


def set_nn_dtype(dtype):
    """Set default dtype for layer paramters"""
    global _NN_DTYPE
    _NN_DTYPE = dtype


def get_nn_dtype():
    return _NN_DTYPE
