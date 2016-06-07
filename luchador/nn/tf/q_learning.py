from __future__ import division

import logging

import tensorflow as tf

_LG = logging.getLogger(__name__)


class QLearningInterface(object):
    """Class for building ops for Q-learning from TFModel"""
    def __init__(self, clip_delta_min=None, clip_delta_max=None):
        self.clip_delta_min = clip_delta_min
        self.clip_delta_max = clip_delta_max

        self.discount_rate = None
        self.pre_states = None
        self.actions = None
        self.rewards = None
        self.post_states = None
        self.continuation = None
        self.error = None
        self.sync_ops = None

    def build(self, q_network):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        input_shape = q_network.input_tensor.get_shape()
        output_shape = q_network.output_tensor.get_shape()
        n_actions = output_shape[1]

        self.discount_rate = tf.placeholder(tf.float32, None, 'discount_rate')
        self.pre_states = tf.placeholder(tf.float32, input_shape, 'pre_state')
        self.actions = tf.placeholder(tf.int64, (None,), 'actions')
        self.rewards = tf.placeholder(tf.float32, (None,), 'rewards')
        self.post_states = tf.placeholder(tf.float32, input_shape, 'post_state')
        self.continuation = tf.placeholder(tf.float32, (None,), 'continuation')

        pre_q_network = q_network.copy()
        post_q_network = pre_q_network.copy()
        # Create a copy of Q network, sharing the original parameter Variables
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            with tf.name_scope('pre_action_q_network'):
                q0 = pre_q_network(self.pre_states)
        # Duplicate the network, without sharing Variables
        with tf.variable_scope('post_action_q_network'):
            q1 = post_q_network(self.post_states)
        # Compute Q value from pre-action state and action from record
        with tf.variable_scope('current_q'):
            mask = tf.one_hot(
                self.actions, n_actions, on_value=1.0, off_value=0.0)
            q0 = tf.mul(q0, mask, name='q_masked')
            q0 = tf.reduce_sum(q0, reduction_indices=1, name='q_observed')
        # Compute max Q value from post-action state and post_q_network,
        # and mask with continuation flag
        with tf.variable_scope('future_q'):
            q1 = tf.reduce_max(q1, reduction_indices=1, name='q_max')
            q1 = tf.mul(q1, self.continuation, name='q_masked')
        with tf.variable_scope('optimal_q'):
            q1 = tf.mul(q1, self.discount_rate, name='q_discounted')
            q_opt = tf.add(q1, self.rewards, name='q_optimal')
        # Compute error between Q value from pre-action state and
        # optimal Q value from post-action state
        with tf.variable_scope('error'):
            q_opt = tf.stop_gradient(q_opt)
            delta = tf.sub(q_opt, q0, name='delta')
            delta = tf.clip_by_value(delta, self.clip_delta_min,
                                     self.clip_delta_max, name='clipped_delta')
            self.error = tf.reduce_mean(tf.square(delta), name='MSE')
        # Create operations for syncing post-action Q network with
        # pre-action Q network
        with tf.variable_scope('sync'):
            sync_ops = []
            params0 = pre_q_network.get_parameter_variables()
            params1 = post_q_network.get_parameter_variables()
            for var0, var1 in zip(params0, params1):
                sync_ops.append(var1.assign(var0))
            self.sync_ops = tf.group(*sync_ops)
        return self
