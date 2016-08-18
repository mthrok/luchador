from __future__ import absolute_import

import unittest

from luchador.nn import (
    Dense,
    Conv2D,
    ReLU,
    Flatten,
    TrueDiv,
)

PARAMETERIZED_LAYER_CLASSES = [
    Dense,
    Conv2D,
    TrueDiv,
]

UNPARAMETERIZED_LAYER_CLASSES = [
    Flatten,
    ReLU,
]


ARGS1 = {
    'Dense': {
        'n_nodes': 100,
    },
    'Conv2D': {
        'filter_height': 8,
        'filter_width': 8,
        'n_filters': 32,
        'strides': 1,
        'padding': 'VALID',
    },
    'TrueDiv': {
        'denom': 1,
    }
}

ARGS2 = {
    'Dense': {
        'n_nodes': 200,
    },
    'Conv2D': {
        'filter_height': 4,
        'filter_width': 4,
        'n_filters': 16,
        'strides': 2,
        'padding': 'SAME',
    },
    'TrueDiv': {
        'denom': 255
    }
}


def make_parameterized_layers(args):
    return [Layer(**args[Layer.__name__])
            for Layer in PARAMETERIZED_LAYER_CLASSES]


def make_un_parameterized_layers():
    return [Layer() for Layer in UNPARAMETERIZED_LAYER_CLASSES]


class LayerTest(unittest.TestCase):
    def test_parameterized_layer_equality(self):
        """Parameterized layers with same arguments must be equal"""
        layers1 = make_parameterized_layers(ARGS1)
        layers2 = make_parameterized_layers(ARGS1)
        for l1, l2 in zip(layers1, layers2):
            self.assertEqual(
                l1, l2,
                '{} initialized with the '
                'same arguments must be equal.'.format(type(l1)))

    def test_parameterized_layer_inequality(self):
        """Parameterized layers with different arguments must not be equal"""
        layers1 = make_parameterized_layers(ARGS1)
        layers2 = make_parameterized_layers(ARGS2)
        for l1, l2 in zip(layers1, layers2):
            self.assertNotEqual(
                l1, l2,
                '{} initialized with the '
                'differnet arguments must not be equal.'.format(type(l1)))

    def test_un_parameterized_layer_equality(self):
        """Instances of not parmaeterized layers must be equal"""
        layers1 = make_un_parameterized_layers()
        layers2 = make_un_parameterized_layers()
        for l1, l2 in zip(layers1, layers2):
            self.assertEqual(
                l1, l2, 'Instances of {} must be eqauel'.format(type(l1)))
