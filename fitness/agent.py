import logging

_LG = logging.getLogger(__name__)


class _Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        raise NotImplementedError('act method is not implemented')

    def reset(self):
        pass


class RandomAgent(_Agent):
    def __init__(self, action_space):
        super(RandomAgent, self).__init__(action_space)

    def act(self, observation, reward=None, done=False):
        return self.action_space.sample()
