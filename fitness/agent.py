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

    def observe(self, action, observation, reward, done, info):
        """Observe the action and it's outcome.

        Args:
          action: The action that this agent previously took.
          observation: Observation (of environment) caused by the action.
          reward: Reward acquired by the action.
          done (bool): Indicates if a task is complete or not.
          info (dict): Infomation related to environment.

          observation, reward, done, info are variables returned by
          environment. See gym.core:Env.step.
        """
        pass

    def act(self):
        """Choose action. Must be implemented in subclass."""
        raise NotImplementedError('act method is not implemented.')

    def reset(self, observation):
        """Reset agent with the initial state of the environment."""
        pass


class RandomAgent(_Agent):
    def __init__(self, action_space, observation_space=None, **kwargs):
        super(RandomAgent, self).__init__(
            action_space=action_space, observation_space=observation_space)

    def act(self):
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

    def reset(self, observation):
        self.reset_history()
        self.observation_history = [observation]

    def initial_q_value(self):
        mean, std = self.config['init_mean'], self.config['init_std']
        return mean + std * np.random.randn(self.action_space.n)

    def observe(self, action, observation, reward, done, info):
        self.action_history.append(action)
        self.observation_history.append(observation)
        self.reward_history.append(reward)
        self.done = done

    def learn(self):
        if len(self.observation_history) < 2:
            return

        src = self.observation_history[-2]
        act = self.action_history[-1]
        rew = self.reward_history[-1]
        tgt = self.observation_history[-1]

        future = 0.0 if self.done else np.max(self.q[tgt])
        lr, beta = self.config['learning_rate'], self.config['discount']
        self.q[src][act] -= lr * (self.q[src][act] - rew - beta * future)

    def choose_action(self):
        last_observation = self.observation_history[-1]
        if np.random.random() > self.config['eps']:
            action = np.argmax(self.q[last_observation])
        else:
            action = self.action_space.sample()
        return action

    def act(self):
        """Select Action based on Epsilon-greedy policy"""
        self.learn()
        self.choose_action()
