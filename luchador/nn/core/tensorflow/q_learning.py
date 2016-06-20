from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ..base.q_learning import QLearningInterface as BaseQLI
from .tensor import Input

_LG = logging.getLogger(__name__)


class QLearningInterface(BaseQLI):
    def get_inputs(self):
        return {
            'pre_states': self.pre_states.tensor,
            'actions': self.actions.tensor,
            'rewards': self.rewards.tensor,
            'post_states': self.post_states.tensor,
            'continuations': self.continuations.tensor,
        }

    def build(self, q_network):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        input_shape = q_network.input.get_shape()
        output_shape = q_network.output.get_shape()
        n_actions = output_shape[1]

        self.pre_states = Input(dtype=tf.uint8, shape=input_shape, name='pre_states')()
        self.actions = Input(dtype=tf.int64, shape=(None,), name='actions')()
        self.rewards = Input(dtype=tf.float32, shape=(None,), name='rewards')()
        self.post_states = Input(dtype=tf.uint8, shape=input_shape, name='post_states')()
        self.continuations = Input(dtype=tf.float32, shape=(None,), name='continuations')()

        pre_q_network = q_network.copy()
        post_q_network = pre_q_network.copy()
        # Create a copy of Q network, sharing the original parameter Variables
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            with tf.name_scope('pre_action_q_network'):
                q0 = pre_q_network(self.pre_states).tensor
        # Duplicate the network, without sharing Variables
        with tf.variable_scope('post_action_q_network'):
            q1 = post_q_network(self.post_states).tensor
        # Compute Q value from pre-action state and action from record
        with tf.variable_scope('current_q'):
            mask = tf.one_hot(
                self.actions.tensor, n_actions, on_value=1.0, off_value=0.0)
            q0 = tf.mul(q0, mask, name='q_masked')
            q0 = tf.reduce_sum(q0, reduction_indices=1, name='q_observed')
        # Compute max Q value from post-action state and post_q_network,
        # and mask with continuation flag
        with tf.variable_scope('future_q'):
            q1 = tf.reduce_max(q1, reduction_indices=1, name='q_max')
            q1 = tf.mul(q1, self.continuations.tensor, name='q_masked')
        with tf.variable_scope('optimal_q'):
            q1 = tf.mul(q1, self.discount_rate, name='q_discounted')
            q_opt = tf.add(q1, self.rewards.tensor, name='q_optimal')
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
            for key in params0.keys():
                var0 = params0[key]
                var1 = params1[key]
                sync_ops.append(var1.assign(var0))
            self.sync_ops = tf.group(*sync_ops)
        return self
