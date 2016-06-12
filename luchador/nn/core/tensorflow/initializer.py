from __future__ import absolute_import

from tensorflow import (  # nopep8
    constant_initializer as Constant,
    random_normal_initializer as Normal,
    random_uniform_initializer as Uniform,
)
from tensorflow.contrib.layers import (  # nopep8
    xavier_initializer as Xavier,
    xavier_initializer_conv2d as XavierConv2D,
)
