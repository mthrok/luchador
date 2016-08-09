from __future__ import absolute_import

import warnings

_NN_BACKEND = 'tensorflow'
_NN_CONV_FORMAT = 'NCHW'
_NN_DTYPE = 'float32'

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


def set_nn_conv_format(cnn_format):
    """Set default convolution data format for Tensorflow"""
    if _NN_BACKEND == 'theano':
        warnings.warn('cnn format only affects Tensorflow backend.')
        return

    if cnn_format not in ['NCHW', 'NHWC']:
        raise ValueError('cnn_format must be either `NCHW` or `NHWC`')

    global _NN_CONV_FORMAT
    _NN_CONV_FORMAT = cnn_format


def get_nn_conv_format():
    return _NN_CONV_FORMAT


def set_nn_dtype(dtype):
    """Set default dtype for layer paramters"""
    global _NN_DTYPE
    _NN_DTYPE = dtype


def get_nn_dtype():
    return _NN_DTYPE
