from __future__ import absolute_import

import logging

from gym import spaces

from luchador.core import Agent

__all__ = ['RandomAgent', 'ControllerAgent']

_LG = logging.getLogger(__name__)


class RandomAgent(Agent):
    def __init__(self, env, agent_config, global_config):
        super(RandomAgent, self).__init__(
            env, agent_config=agent_config, global_config=global_config)

    def reset(self, observation):
        pass

    def observe(self, action, observation, reward, done, info):
        pass

    def act(self):
        return self.action_space.sample()


class ControllerAgent(Agent):
    # TODO: Add game pad controll
    def __init__(self, env, agent_config, global_config):
        if type(env.action_space) not in [spaces.Discrete, spaces.Tuple]:
            raise NotImplementedError(
                'Only Discrete and Tuple spaces are supported now.')

        super(ControllerAgent, self).__init__(
            env, agent_config=agent_config, global_config=global_config)

    def reset(self, observation):
        pass

    def observe(self, action, observation, reward, done, info):
        pass

    def _parse_action(self, action_space):
        if isinstance(action_space, spaces.Discrete):
            _LG.info('Input value: [0, {}]'.format(action_space.n-1))
            return int(raw_input())
        else:
            ret = []
            _LG.info('Input {} inputs'.format(len(action_space.spaces)))
            for space in action_space.spaces:
                ret.append(self._parse_action(space))
            return ret

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
