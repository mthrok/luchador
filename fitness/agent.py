import logging
from collections import defaultdict

import numpy as np
from gym import spaces

from fitness.error import UnsupportedSpace

_LG = logging.getLogger(__name__)


class _Agent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation, reward, done):
        raise NotImplementedError('act method is not implemented')

    def reset(self):
        pass


class RandomAgent(_Agent):
    def __init__(self, action_space, observation_space=None, **kwargs):
        super(RandomAgent, self).__init__(
            action_space=action_space, observation_space=observation_space)

    def act(self, observation, reward=None, done=False):
        return self.action_space.sample()


class ControllerAgent(_Agent):
    # TODO: Add game pad controll
    def __init__(self, action_space, observation_space):
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

    def act(self, observation, reward, done):
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


class TabularQAgent(_Agent):
    """TabularQAgent from gym example"""
    def __init__(self, action_space, observation_space, **userconfig):
        # TODO: Make this work for Tuple(Descrete...) types
        # Currently work with Roulette (but the result is not impressive)
        if not isinstance(observation_space, spaces.Discrete):
            raise UnsupportedSpace(
                'Observation space {} incompatible with {}. '
                '(Only supports Discrete observation spaces.)'
                .format(observation_space, self))
        if not isinstance(action_space, spaces.Discrete):
            raise UnsupportedSpace(
                'Action space {} incompatible with {}. '
                '(Only supports Discrete action spaces.)'
                .format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = {
            'init_mean': 0.0,       # Initialize Q values with this mean
            'init_std': 0.0,        # Initialize Q values with this standard deviation
            'learning_rate': 0.1,
            'eps': 0.05,            # Epsilon in epsilon greedy policies
            'discount': 0.95,
            'n_iter': 10000,        # Number of iterations
        }
        self.config.update(userconfig)
        self.q = defaultdict(lambda: self.initial_q_value())
        self.reset_history()

    def reset_history(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.done = False

    def reset(self):
        self.state = []

    def initial_q_value(self):
        mean, std = self.config['init_mean'], self.config['init_std']
        return mean + std * np.random.randn(self.action_space.n)

    def choose_action(self, observation, eps=None):
        """Select Action based on Epsilon-greedy policy"""
        eps = self.config['eps'] if eps is None else eps
        if np.random.random() > eps:
            action = np.argmax(self.q[observation])
        else:
            action = self.action_space.sample()
        return action

    def act(self, observation, reward, done):
        self.observation_history.append(observation)
        self.reward_history.append(reward)
        self.done = done
        self.learn()
        action = self.choose_action(observation)
        self.action_history.append(action)
        return action

    def learn(self):
        if len(self.observation_history) < 2:
            return

        obs2, obs1 = self.observation_history[-2:]
        act1 = self.action_history[-1]
        reward1 = self.reward_history[-1]

        future = 0.0 if self.done else np.max(self.q[obs1])
        lr, beta = self.config['learning_rate'], self.config['discount']
        self.q[obs2][act1] -= lr * (self.q[obs2][act1] - reward1 - beta * future)
