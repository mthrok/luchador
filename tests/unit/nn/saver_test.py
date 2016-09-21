from __future__ import absolute_import

import unittest

import os
import h5py
import numpy as np

from luchador.nn.saver import Saver

FILE_PATH = 'foo.dat'


class SaverTest(unittest.TestCase):
    def setUp(self):
        if os.path.exists(FILE_PATH):
            os.remove(FILE_PATH)

    def test_save(self):
        """`save` function can save data"""
        key = 'test'
        value1 = np.random.randint(low=0, high=2, size=(32, 8, 8, 4))

        saver = Saver(FILE_PATH)
        saver.save({key: value1})

        f = h5py.File(FILE_PATH)
        value2 = f[key]
        self.assertTrue(
            np.all(value1 == value2),
            'Saved value do not match. '
            'Expected: {}. Found: {}'.format(value1, value2)
        )

    def test_overwrite(self):
        """`save` function appends to existing file"""
        key1 = 'test1'
        value1 = np.random.randint(low=0, high=2, size=(32, 8, 8, 4))
        saver1 = Saver(FILE_PATH)
        saver1.save({key1: value1})

        key2 = 'test2'
        value2 = np.random.randint(low=0, high=2, size=(32, 8, 8, 4))
        saver2 = Saver(FILE_PATH)
        saver2.save({key2: value2})

        f = h5py.File(FILE_PATH)
        value2_ = f[key2]
        self.assertTrue(
            np.all(value2 == value2_),
            'Saved value do not match. '
            'Expected: {}. Found: {}'.format(value1, value2)
        )
