from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ..base import DeepQLearning as BaseQLI
from . import config as CFG
from .tensor import Tensor, Input

__all__ = ['DeepQLearning']

_LG = logging.getLogger(__name__)


class DeepQLearning(BaseQLI):
    def build(self, model_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        with tf.variable_scope('pre_trans'):
            self.pre_trans_model = model_maker()
            self.pre_states = self.pre_trans_model.input
        with tf.variable_scope('post_trans'):
            self.post_trans_model = model_maker()
            self.post_states = self.post_trans_model.input
        with tf.variable_scope('target_q_value'):
            self._build_target_q_value()
        with tf.variable_scope('sync'):
            self._build_sync_ops()
        return self

    def _build_target_q_value(self):
        self.actions = Input(dtype='int32', shape=(None,), name='actions')
        self.rewards = Input(dtype=CFG.DTYPE, shape=(None,), name='rewards')
        self.continuations = Input(
            dtype=CFG.DTYPE, shape=(None,), name='continuations')

        actions = self.actions().tensor
        rewards = self.rewards().tensor
        continuations = self.continuations().tensor
        with tf.name_scope('future_reward'):
            with tf.name_scope('future_q_value'):
                post_q = self.post_trans_model.output.tensor
                post_q = tf.reduce_max(
                    post_q, reduction_indices=1, name='max_post_q')
                post_q = tf.mul(
                    post_q, self.discount_rate, name='discounted_max_post_q')
                post_q = tf.mul(
                    post_q, continuations, name='masked_max_post_q')

            if self.min_reward and self.max_reward:
                with tf.name_scope('clipped_reward'):
                    rewards = tf.clip_by_value(
                        rewards, self.min_reward, self.max_reward)

            future = tf.add(rewards, post_q, name='future_reward')

        with tf.name_scope('target_q_value'):
            n_actions = self.pre_trans_model.output.get_shape()[1]
            with tf.name_scope('reshape_current_q_value'):
                mask_off = tf.one_hot(actions, depth=n_actions, on_value=0.,
                                      off_value=1., name='actions_not_taken')
                current = tf.identity(self.pre_trans_model.output.tensor)
                current = current * mask_off

            with tf.name_scope('reshape_future_q_value'):
                mask_on = tf.one_hot(actions, depth=n_actions, on_value=1.,
                                     off_value=0., name='actions_taken')
                future = tf.reshape(future, (-1, 1))
                future = tf.tile(future, tf.pack([1, n_actions]))
                future = future * mask_on

            target_q = tf.add(current, future, name='target_q')
        self.target_q = Tensor(tensor=target_q)

    def _build_sync_ops(self):
        src_vars = self.pre_trans_model.get_parameter_variables().values()
        tgt_vars = self.post_trans_model.get_parameter_variables().values()
        ops = [tgt.tensor.assign(src.tensor)
               for src, tgt in zip(src_vars, tgt_vars)]
        self.sync_op = tf.group(*ops, name='sync')
