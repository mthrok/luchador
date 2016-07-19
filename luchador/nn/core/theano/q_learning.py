from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

import theano.tensor as T

from ..base import DeepQLearning as BaseQLI
from . import scope as scp
from .tensor import Input, Tensor, Operation

_LG = logging.getLogger(__name__)


class DeepQLearning(BaseQLI):
    def build(self, model_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        with scp.variable_scope('pre_trans'):
            self.pre_trans_model = model_maker()
            self.pre_states = self.pre_trans_model.input
        with scp.variable_scope('post_trans'):
            self.post_trans_model = model_maker()
            self.post_states = self.post_trans_model.input
        with scp.variable_scope('target_q_value'):
            self._build_target_q_value()
        with scp.variable_scope('sync'):
            self._build_sync_op()
        return self

    def _build_target_q_value(self):
        self.actions = Input(dtype='uint8', shape=(None,), name='actions')
        self.rewards = Input(dtype='float64', shape=(None,), name='rewards')
        self.terminals = Input(shape=(None,), name='continuations')

        actions = self.actions().tensor
        rewards = self.rewards().tensor
        terminals = self.terminals().tensor

        post_q = self.post_trans_model.output.tensor
        post_q = T.max(post_q, axis=1)
        post_q = post_q * self.discount_rate
        post_q = post_q * (1.0 - terminals)

        if self.min_reward and self.max_reward:
            rewards = rewards.clip(self.min_reward, self.max_reward)
        future = rewards + post_q

        n_actions = self.pre_trans_model.output.get_shape()[1]
        mask_on = T.extra_ops.to_one_hot(actions, n_actions)
        mask_off = 1.0 - mask_on
        current = self.pre_trans_model.output.tensor
        current = current * mask_off

        future = T.reshape(future, (-1, 1))
        future = T.tile(future, [1, n_actions])
        future = future * mask_on

        target_q = current + future
        # TODO: Add shape inference
        self.future_reward = Tensor(tensor=future, shape=(-1, ))
        self.target_q = Tensor(tensor=target_q, shape=(-1, n_actions))

    def _build_sync_op(self):
        sync_op = OrderedDict()
        src_vars = self.pre_trans_model.get_parameter_variables().values()
        tgt_vars = self.post_trans_model.get_parameter_variables().values()
        for src, tgt in zip(src_vars, tgt_vars):
            sync_op[tgt.tensor] = src.tensor
        self.sync_op = Operation(op=sync_op)
