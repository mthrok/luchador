from __future__ import division
from __future__ import absolute_import

import numpy as np
from numpy.random import RandomState

from theano import config

from .random import get_rng


class Initializer(object):
    def __call__(self, shape):
        self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError('`sample` method is not implemented')


class Constant(Initializer):
    def __init__(self, value, dtype=config.floatX):
        self.value = value
        self.dtype = dtype

    def sample(self, shape):
        return self.value * np.ones(shape, dtype=self.dtype)


class Uniform(Initializer):
    def __init__(self, minval=0.0, maxval=1.0, seed=None, dtype=config.floatX):
        self.low = minval
        self.high = maxval
        self.dtype = dtype
        self._rng = RandomState(seed) if seed else get_rng()

    def sample(self, shape):
        values = self._rng.uniform(low=self.low, high=self.high, size=shape)
        return values.astype(self.dtype)


class Normal(Initializer):
    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=config.floatX):
        self.loc = mean
        self.scale = stddev
        self.dtype = dtype
        self._rng = RandomState(seed) if seed else get_rng()

    def sample(self, shape):
        values = self._rng.normal(loc=self.loc, scale=self.scale, size=shape)
        return values.astype(self.dtype)


class Xavier(Initializer):
    """Adoptation of xavier_initializer from tensorflow"""
    def __init__(self, uniform=True, seed=None, dtype=config.floatX):
        self.uniform = uniform
        self.dtype = dtype
        self._rng = RandomState(seed) if seed else get_rng()

    def _compute_param(self, fan_in, fan_out):
        if self.uniform:
            x = np.sqrt(6. / (fan_in + fan_out))
            return {'low': -x, 'high': x}
        else:
            scale = np.sqrt(3. / (fan_in + fan_out))
            return {'loc': 0., 'scale': scale}

    def _sample(self, shape, param):
        if self.uniform:
            values = self._rng.uniform(
                low=param['low'], high=param['high'], size=shape)
        else:
            values = self._rng.normal(
                loc=param['loc'], scale=param['scale'], size=shape)
        return values.astype(self.dtype)

    def sample(self, shape):
        if not len(shape) == 2:
            raise ValueError(
                'Xavier Initializer expects the shape to have 2 elements, '
                'e.g. [fan_in, fan_out]. Found: {}'.format(shape)
            )

        fan_in, fan_out = shape[:2]
        param = self._compute_param(fan_in, fan_out)
        return self._sample(shape, param)


class XavierConv2D(Xavier):
    """Adoptation of xavier_initializer_conv2d from tensorflow"""
    def __init__(self, uniform=True, seed=None, dtype=config.floatX):
        super(XavierConv2D, self).__init__(uniform, seed, dtype)

    def sample(self, shape):
        # theano's filter shape is
        # (output_channels, input_channels, filter_rows, filter_columns)
        fan_in = shape[1] * shape[2] * shape[3]
        fan_out = shape[0] * shape[2] * shape[3]
        param = self._compute_param(fan_in, fan_out)
        return self._sample(shape, param)
