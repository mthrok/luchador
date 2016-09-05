from __future__ import absolute_import

import logging

from luchador.common import StoreMixin

__all__ = ['DeepQLearning']

_LG = logging.getLogger(__name__)


class DeepQLearning(StoreMixin, object):
    """Build Q-learning network and optimization operations"""
    def __init__(self, discount_rate, min_reward=None, max_reward=None):
        self._store_args(
            discount_rate=discount_rate,
            min_reward=min_reward,
            max_reward=max_reward,
        )

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

    def _validate_args(self, args):
        if (
                (args['min_reward'] or args['max_reward']) and
                (not args['max_reward'] or not args['min_reward'])
        ):
            raise ValueError(
                'When clipping reward, both `min_reward` '
                'and `max_reward` must be provided.')

    def __call__(self, q_network_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network_maker(function): Model factory function which are called
            without any arguments and return Model object
        """
        self.build(q_network_maker)

    def build(self, q_network):
        """Build computation graph (error and sync ops) for Q learning"""
        raise NotImplementedError(
            '`build` method is not implemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )
