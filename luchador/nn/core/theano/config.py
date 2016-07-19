from __future__ import absolute_import

import theano

__all__ = ['set_cnn_format', 'set_dtype']


# Following functions are defined for consitency of theano/tensorflow
# compatibility, and there is no plactical use.

def set_cnn_format(cnn_format):
    if not cnn_format == 'NCHW':
        raise ValueError('cnn_format must be `NCHW` in theano backend')


def set_dtype(dtype):
    theano.config.floatX = dtype
