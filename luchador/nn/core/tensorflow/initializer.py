from __future__ import absolute_import

from tensorflow import (  # noqa: F401
    constant_initializer as Constant,
    random_normal_initializer as Normal,
    random_uniform_initializer as Uniform,
)
from tensorflow.contrib.layers import (  # noqa: F401
    xavier_initializer as Xavier,
    xavier_initializer_conv2d as XavierConv2D,
)
