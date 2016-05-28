from __future__ import division

import logging

import tensorflow as tf

_LG = logging.getLogger(__name__)


class QLearningInterface(object):
    def __init__(self, discount_rate, clip_error_norm, clip_grad_norm):
        self.discount_rate = discount_rate
        self.clip_error_norm = clip_error_norm
        self.clip_grad_norm = clip_grad_norm

    def init(self, q_network, session, optimizer):
        self.session = session
        self.optimizer = optimizer
        self._build_training_model(q_network)

    def _build_training_model(self, q_network):
        input_shape = q_network.input_tensor.get_shape()
        output_shape = q_network.output_tensor.get_shape()
        n_actions = output_shape[1]

        self.state0 = tf.placeholder(tf.float32, input_shape, 'state0')
        self.state1 = tf.placeholder(tf.float32, input_shape, 'state1')
        self.action0 = tf.placeholder(tf.int64, (None,), 'action0')
        self.reward0 = tf.placeholder(tf.float32, (None,), 'reward0')
        self.continuation = tf.placeholder(tf.float32, (None,), 'continuation')

        # Create a copy of Q network, sharing Variables
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            with tf.name_scope('QNetwork0'):
                self.q_network0 = q_network.copy()
                q0 = self.q_network0(self.state0)
        # Duplicate network, without sharing Variables
        with tf.variable_scope('QNetwork1'):
            self.q_network1 = self.q_network0.copy()
            q1 = self.q_network1(self.state1)

        with tf.variable_scope('q0'):
            mask = tf.one_hot(
                self.action0, n_actions, on_value=1.0, off_value=0.0)
            q0 = tf.mul(q0, mask, name='q0_masked')
            q0 = tf.reduce_sum(q0, reduction_indices=1, name='q0_observed')

        with tf.variable_scope('q1'):
            q1 = tf.reduce_max(q1, reduction_indices=1, name='q1_max')
            q1 = tf.mul(q1, self.continuation, name='q1_masked')

        with tf.variable_scope('Q'):
            disc = tf.constant(
                self.discount_rate, dtype=tf.float32, name='discount_rate')
            q1 = tf.mul(q1, disc, name='q1_discounted')
            Q = tf.add(q1, self.reward0, name='Q')

        with tf.variable_scope('error'):
            self.error = tf.reduce_mean(
                tf.square(q0 - tf.stop_gradient(Q)), name='MSE')
            if self.clip_error_norm:
                self.error = tf.clip_by_norm(
                    self.error, self.clip_error_norm, name='MSE_clipped')

        with tf.variable_scope('training'):
            grads = self.optimizer.compute_gradients(self.error)
            if self.clip_grad_norm:
                for i, (grad, var) in enumerate(grads):
                    if grad is None:
                        continue
                    grad = tf.clip_by_norm(
                        grad, self.clip_grad_norm, name='gradient_clipped')
                    grads[i] = (grad, var)
            self.train = self.optimizer.apply_gradients(grads)

        with tf.variable_scope('sync'):
            sync_ops = []
            params0 = self.q_network0.layer_parameters
            params1 = self.q_network1.layer_parameters
            for key in params0.keys():
                sync_ops.append(params1[key].assign(params0[key]))
            self.sync_ops = tf.group(*sync_ops)

    def train_network(self, state0, action0, reward0, state1, continuation):
        return self.session.run(
            [self.error, self.train],
            feed_dict={
                self.state0: state0,
                self.action0: action0,
                self.reward0: reward0,
                self.state1: state1,
                self.continuation: continuation,
            }
        )[0]

    def sync_networks(self):
        self.session.run(self.sync_ops)
