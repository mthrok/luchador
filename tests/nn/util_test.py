from __future__ import absolute_import

import unittest

from tests.fixture import (
    get_layers,
    get_optimizers,
    get_initializers,
)

from luchador.nn.util import (
    get_layer,
    get_optimizer,
    get_initializer,
)

INITIALIZERS = get_initializers()
N_INITIALIZERS = 5

OPTIMIZERS = get_optimizers()
N_OPTIMIZERS = 4

LAYERS = get_layers()
N_LAYERS = 5


class UtilTest(unittest.TestCase):
    def test_initializer_test_coverage(self):
        """All initializers are tested"""
        self.assertEqual(
            len(INITIALIZERS), N_INITIALIZERS,
            'Number of initializers are changed. (New initializer is added?) '
            'Fix unittest to cover new initializers'
        )

    def test_optimizer_test_coverage(self):
        """All initializers are tested"""
        self.assertEqual(
            len(OPTIMIZERS), N_OPTIMIZERS,
            'Number of optimizers are changed. (New optimizer is added?) '
            'Fix unittest to cover new optimizers'
        )

    def test_layer_test_coverage(self):
        """All initializers are tested"""
        self.assertEqual(
            len(LAYERS), N_LAYERS,
            'Number of layers are changed. (New layer is added?) '
            'Fix unittest to cover new layers'
        )

    def test_get_initalizer(self):
        """get_initializer returns correct initalizer class"""
        for name, Initializer in INITIALIZERS.items():
            expected = Initializer
            found = get_initializer(name)
            self.assertEqual(
                expected, found,
                'get_initializer returned wrong initializer Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )

    def test_get_optimzier(self):
        """get_optimizer returns correct optimizer class"""
        for name, Optimizer in OPTIMIZERS.items():
            expected = Optimizer
            found = get_optimizer(name)
            self.assertEqual(
                expected, found,
                'get_optimizer returned wrong optimizer Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )

    def test_get_layer(self):
        """get_layer returns correct layer class"""
        for name, Layer in LAYERS.items():
            expected = Layer
            found = get_layer(name)
            self.assertEqual(
                expected, found,
                'get_layer returned wrong layer Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )
