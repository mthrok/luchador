from __future__ import absolute_import

import logging

import numpy as np

from .base import Agent as BaseAgent

__all__ = ['RandomAgent']

_LG = logging.getLogger(__name__)


class RandomAgent(BaseAgent):
    def __init__(self, env, agent_config, global_config):
        super(RandomAgent, self).__init__(
            env, agent_config=agent_config, global_config=global_config)

    def reset(self, observation):
        pass

    def observe(self, action, observation, reward, done, info):
        pass

    def act(self):
        return np.random.randint(self.env.n_actions)
