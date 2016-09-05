from __future__ import absolute_import

from luchador.common import get_subclasses

__all__ = ['BaseAgent', 'get_agent']


class BaseAgent(object):
    def set_env_info(self, env):
        """Retrieve environmental information"""
        pass

    def init(self):
        pass

    def observe(self, action, observation, reward, terminal, env_state):
        """Observe the action and it's outcome.

        Args:
          action: The action that this agent previously took.
          observation: Observation (of environment) caused by the action.
          reward: Reward acquired by the action.
          done (bool): Indicates if a task is complete or not.
          env_state (dict): Infomation related to environment.

        observation, reward, done, info are variables returned by
          environment. See gym.core:Env.step.
        """
        raise NotImplementedError('observe method is not implemented.')

    def act(self):
        """Choose action. Must be implemented in subclass."""
        raise NotImplementedError('act method is not implemented.')

    def reset(self, observation):
        """Reset agent with the initial state of the environment."""
        raise NotImplementedError('reset method is not implemented.')

    def perform_post_episode_task(self):
        """Perform post episode task"""
        pass


def get_agent(name):
    for Class in get_subclasses(BaseAgent):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Agent: {}'.format(name))
