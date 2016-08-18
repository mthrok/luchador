from __future__ import absolute_import

import unittest

from luchador.nn import (
    Model,
    Dense,
    Conv2D,
)


def make_layers():
    conv = Conv2D(filter_height=8, filter_width=8, n_filters=8, strides=4)
    dense1 = Dense(n_nodes=512)
    dense2 = Dense(n_nodes=256)
    return conv, dense1, dense2


class ModelTest(unittest.TestCase):
    def test_model_equality(self):
        """Models with the same configuration should be euqal"""
        m1, m2 = Model(), Model()

        conv, dense, _ = make_layers()

        m1.add_layer(conv, scope='conv')
        m2.add_layer(conv, scope='conv')

        m1.add_layer(dense, scope='dense')
        m2.add_layer(dense, scope='dense')

        self.assertEqual(
            m1, m2, 'Models with the same configuration must be equal')

    def test_model_equality_empty(self):
        """Models without layers should be euqal"""
        m1, m2 = Model(), Model()
        self.assertEqual(m1, m2, 'Models without layers must be equal')

    def test_model_inequality_type(self):
        """Models is not equal with other types"""
        m = Model()

        conv, dense, _ = make_layers()

        m.add_layer(conv, scope='conv')
        m.add_layer(dense, scope='dense')

        self.assertNotEqual(
            m, 1, 'A model instance should not be equal with an int')
        self.assertNotEqual(
            m, {}, 'A model instance should not be equal with a dict')
        self.assertNotEqual(
            m, [], 'A model instance should not be equal with a list')
        self.assertNotEqual(
            m, '', 'A model instance should not be equal with a str')
        self.assertNotEqual(
            m, object(), 'A model instance should not be equal with an object')

    def test_model_inequality_number_of_layers(self):
        """Models with different #layers are not equal"""
        m1, m2 = Model(), Model()

        conv, dense, _ = make_layers()

        m1.add_layer(conv, scope='conv')
        m2.add_layer(conv, scope='conv')

        m1.add_layer(dense, scope='dense')

        self.assertNotEqual(
            m1, m2, 'Models with the different #layers must not be equal')

    def test_model_inequality_layer_types(self):
        """Models with different layers are not equal"""
        m1, m2 = Model(), Model()

        conv, dense, _ = make_layers()

        m1.add_layer(conv, scope='conv')

        m2.add_layer(dense, scope='dense')

        self.assertNotEqual(
            m1, m2, 'Models with the different layers must not be equal')

    def test_model_inequality_layer_configurations(self):
        """Models with different layer configurations are not equal"""
        m1, m2 = Model(), Model()

        _, dense1, dense2 = make_layers()

        m1.add_layer(dense1, scope='dense1')

        m2.add_layer(dense2, scope='dense2')

        self.assertNotEqual(
            m1, m2,
            'Models with the different layer configurations must not be equal')
