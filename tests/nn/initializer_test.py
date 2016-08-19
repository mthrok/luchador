from __future__ import absolute_import

import unittest

from tests.fixture import get_initializers
from luchador.nn.util import get_initializer

INITIALIZERS = get_initializers()
N_INITIALIZERS = 5

ARGS1 = {
    'Constant': {
        'value': 100,
        'dtype': 'float32',
    },
    'Uniform': {
        'minval': 10,
        'maxval': 100,
        'seed': 1,
        'dtype': 'float32',
    },
    'Normal': {
        'mean': 0,
        'stddev': 1,
        'seed': 1,
        'dtype': 'float32',
    },
    'Xavier': {
        'uniform': True,
        'seed': 1,
        'dtype': 'float32',
    },
    'XavierConv2D': {
        'uniform': True,
        'seed': 1,
        'dtype': 'float32',
    }
}

ARGS2 = {
    'Constant': {
        'value': 10,
        'dtype': 'float64',
    },
    'Uniform': {
        'minval': 5,
        'maxval': 50,
        'seed': 10,
        'dtype': 'float64',
    },
    'Normal': {
        'mean': 1,
        'stddev': 10,
        'seed': 10,
        'dtype': 'float64',
    },
    'Xavier': {
        'uniform': False,
        'seed': 10,
        'dtype': 'float64',
    },
    'XavierConv2D': {
        'uniform': False,
        'seed': 10,
        'dtype': 'float64',
    }
}


def make_initializers(args):
    return [Layer(**args[name]) for name, Layer in INITIALIZERS.items()]


class InitializerTest(unittest.TestCase):
    def test_initializer_equality(self):
        """Initializers with same arguments must be equal"""
        initializers1 = make_initializers(ARGS1)
        initializers2 = make_initializers(ARGS1)
        for i1, i2 in zip(initializers1, initializers2):
            self.assertEqual(
                i1, i2,
                '{} initialized with the '
                'same arguments must be equal.'.format(type(i1)))

    def test_initializer_inequality(self):
        """Initializers with different arguments must not be equal"""
        initializers1 = make_initializers(ARGS1)
        initializers2 = make_initializers(ARGS2)
        for i1, i2 in zip(initializers1, initializers2):
            self.assertNotEqual(
                i1, i2,
                '{} initialized with the '
                'differnet arguments must not be equal.'.format(type(i1)))

    def test_initializer_export(self):
        """`export` method should return constructor arguments """
        initializers = make_initializers(ARGS1)
        for initializer in initializers:
            args = initializer.export()
            found = args['args']
            expected = ARGS1[args['name']]
            self.assertEqual(
                found, expected,
                '{} exported arguments different than given in constructor')

    def test_initializer_equality_export(self):
        """Initializers recreated with export are identical to originals"""
        for initializer0 in make_initializers(ARGS1):
            args = initializer0.export()
            initializer1 = get_initializer(args['name'])(**args['args'])
            expected = initializer0
            found = initializer1
            self.assertEqual(
                expected, found,
                'Initializer recreated from export is not identical to '
                'original. Expected: {}, Found: {}'.format(expected, found))
