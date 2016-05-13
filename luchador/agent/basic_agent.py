import logging

from gym import spaces

from luchador.core import Agent
from luchador.error import UnsupportedSpace

__all__ = ['RandomAgent', 'ControllerAgent']

_LG = logging.getLogger(__name__)


class RandomAgent(Agent):
    def __init__(self, action_space, observation_space):
        super(RandomAgent, self).__init__(
            action_space=action_space, observation_space=observation_space)

    def act(self):
        return self.action_space.sample()


class ControllerAgent(Agent):
    # TODO: Add game pad controll
    def __init__(self, action_space, observation_space, **config):
        super(ControllerAgent, self).__init__(
            action_space=action_space, observation_space=observation_space)

    def parse_action(self, action_space):
        if isinstance(action_space, spaces.Discrete):
            _LG.info('Input value: [0, {}]'.format(action_space.n-1))
            return int(raw_input())
        elif isinstance(self.action_space, spaces.Tuple):
            ret = []
            _LG.info('Input {} inputs'.format(len(action_space.spaces)))
            for space in action_space.spaces:
                ret.append(self.parse_action(space))
            return ret
        else:
            raise UnsupportedSpace(
                'Only Discrete and Tuple spaces are supported now.')

    def act(self):
        while True:
            try:
                action = self.parse_action(self.action_space)
            except ValueError:
                _LG.error('Failed to parse. Retry.')
                continue

            if self.action_space.contains(action):
                return action
            _LG.error('Invalid action was given. Retry.')
            _LG.debug(action)
