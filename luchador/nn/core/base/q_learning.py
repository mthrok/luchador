from __future__ import absolute_import

import logging


_LG = logging.getLogger(__name__)


class QLearningInterface(object):
    """Build ops for Q-learning on top of the given Model"""
    def __init__(self, discount_rate, clip_delta_min=None, clip_delta_max=None):
        self.clip_delta_min = clip_delta_min
        self.clip_delta_max = clip_delta_max

        self.discount_rate = discount_rate
        self.pre_states = None
        self.actions = None
        self.rewards = None
        self.post_states = None
        self.continuations = None
        self.error = None
        self.sync_ops = None

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
