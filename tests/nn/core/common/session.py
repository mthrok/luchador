from __future__ import absolute_import

import unittest

import numpy as np

from luchador.nn import (
    scope,
    Session
)


class SessionTest(unittest.TestCase):
    def test_load_dataset(self):
        """Variable value can be set via Session.load_dataset"""
        name = 'test_load_dataset'
        shape = (3, 3)
        dtype = 'float'
        target_value = 10

        variable = scope.get_variable(name=name, shape=shape, dtype=dtype)
        value = target_value * np.ones(shape)

        session = Session()
        session.load_dataset({name: value})

        updated_value = session.run(name=None, outputs=variable)
        self.assertTrue(np.all(target_value == updated_value))
