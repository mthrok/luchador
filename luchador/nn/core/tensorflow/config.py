CNN_FORMAT = 'NCHW'
DTYPE = 'float32'

__all__ = ['set_cnn_format', 'set_dtype']


def set_cnn_format(cnn_format):
    if cnn_format not in ['NCHW', 'NHWC']:
        raise ValueError('cnn_format must be either `NCHW` or `NHWC`')
    global CNN_FORMAT
    CNN_FORMAT = cnn_format


def set_dtype(dtype):
    global DTYPE
    DTYPE = dtype
