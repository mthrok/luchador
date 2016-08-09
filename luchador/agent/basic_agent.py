from __future__ import absolute_import

import logging

from .base import Agent as BaseAgent

__all__ = ['RandomAgent', 'ControllerAgent']

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
        return self.action_space.sample()


class ControllerAgent(BaseAgent):
    # TODO: Add game pad controll
    def __init__(self, env, agent_config, global_config):
        super(ControllerAgent, self).__init__(
            env, agent_config=agent_config, global_config=global_config)

    def reset(self, observation):
        pass

    def observe(self, action, observation, reward, done, info):
        pass

    def _parse_action(self, action_space):
        _LG.info('Input value: [0, {}]'.format(action_space.n-1))
        return int(raw_input())

    def act(self):
        while True:
            try:
                action = self._parse_action(self.action_space)
            except ValueError:
                _LG.error('Failed to parse. Retry.')
                continue

            if self.action_space.contains(action):
                return action
            _LG.error('Invalid action was given. Retry.')
            _LG.debug(action)
