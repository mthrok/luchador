from __future__ import absolute_import

import unittest

from tests.fixture import get_initializers

from luchador.nn.util import (
    get_initializer,
)

INITIALIZERS = get_initializers()
N_INITIALIZERS = 5


class UtilTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(
            len(INITIALIZERS), N_INITIALIZERS,
            'Number of initializers are changed. (New initializer is added?) '
            'Fix unittest to cover new initializers'
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
