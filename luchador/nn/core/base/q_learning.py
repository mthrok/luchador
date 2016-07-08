from __future__ import absolute_import

import logging


_LG = logging.getLogger(__name__)


class DeepQLearning(object):
    """Build ops for Q-learning on top of the given Model"""
    def __init__(self, discount_rate, min_reward=None, max_reward=None):
        self.discount_rate = discount_rate
        self.min_reward = min_reward
        self.max_reward = max_reward

        self.pre_states = None
        self.actions = None
        self.rewards = None
        self.post_states = None
        self.continuations = None

        self.pre_trans_model = None
        self.pre_trans_model = None
        self.sync_op = None

    def build(self, q_network):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        raise NotImplementedError(
            '`build` method is not implemnted for {} class'
            .format(type(self).__name__)
        )
