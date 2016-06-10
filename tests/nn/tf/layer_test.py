import unittest

import tensorflow as tf

from luchador.nn.tf.layer import Dense


def create_input(shape):
    return tf.placeholder(tf.float32, shape=shape)


class TestDense(unittest.TestCase):
    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        tf.reset_default_graph()
        self.session.close()
        self.session = None

    def test_init(self):
        """Dense layer does not crash"""
        n_nodes = 256
        dense = Dense(n_nodes)
        input_tensor = create_input(shape=(None, n_nodes))
        dense(input_tensor)

    def test_copy_success_with_reuse(self):
        """Copied layer can create node when reuse=True in variable scope"""
        n_nodes = 256
        input_tensor = create_input(shape=(None, n_nodes))
        dense1 = Dense(n_nodes)
        dense1(input_tensor)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            dense2 = dense1.copy()
            dense2(input_tensor)

        vars1 = dense1.parameter_variables
        vars2 = dense2.parameter_variables
        expected = set(vars1.keys())
        found = set(vars2.keys())
        self.assertEqual(
            expected, found,
            'Copied layer should have the same parameter variables '
            'as the original layer. Expected: {}, Found: {}'
            .format(expected, found))
        for key in vars1.keys():
            self.assertTrue(
                vars1[key] is vars2[key],
                'Variable objects in copied layer should be identical '
                'to those in the original layer. Variable {} is not identical.'
                .format(key)
            )

    def test_copy_fail_without_reuse(self):
        """Copied layer fails to create node when reuse is not True"""
        n_nodes = 256
        input_tensor = create_input(shape=(None, n_nodes))
        dense1 = Dense(n_nodes)
        dense1(input_tensor)
        try:
            dense3 = dense1.copy()
            dense3(input_tensor)
            self.fail('Copied layer should raise ValueError when '
                      'reuse is not enabled in variable scope.')
        except ValueError:
            pass
        except Exception:
            self.fail('It should be ValueError when copied layer tries to '
                      'create node without reuse enabled in variable scope.')
