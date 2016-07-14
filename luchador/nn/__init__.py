from __future__ import absolute_import

CNN_FORMAT = 'NHWC'
DTYPE = 'float32'


def set_cnn_format(cnn_format):
    if cnn_format not in ['NCHW', 'NHWC']:
        raise ValueError('cnn_format must be either `NCHW` or `NHWC`')
    global CNN_FORMAT
    CNN_FORMAT = cnn_format


def set_dtyp(dtype):
    global DTYPE
    DTYPE = dtype


###############################################################################
from .core import *  # nopep8
