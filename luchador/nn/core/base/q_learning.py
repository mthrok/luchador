from __future__ import absolute_import

import logging

__all__ = ['DeepQLearning']

_LG = logging.getLogger(__name__)


class DeepQLearning(object):
    """Build Q-learning network and optimization operations"""
    def __init__(self, discount_rate, min_reward=None, max_reward=None):
        if (
                (min_reward is None and max_reward is not None) or
                (max_reward is None and min_reward is not None)
        ):
            raise ValueError(
                'When clipping reward, both `min_reward` '
                'and `max_reward` must be provided.')

        # Paramters
        self.discount_rate = discount_rate
        self.min_reward = min_reward
        self.max_reward = max_reward

        # Inputs to the network
        self.pre_states = None
        self.actions = None
        self.rewards = None
        self.post_states = None
        self.terminals = None

        # Actual NN models
        self.pre_trans_net = None
        self.pre_trans_net = None

        # Q values
        self.predicted_q = None
        self.target_q = None

        self.sync_op = None

    def build(self, q_network):
        """Build computation graph (error and sync ops) for Q learning"""
        raise NotImplementedError(
            '`build` method is not implemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )
