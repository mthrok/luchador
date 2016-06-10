from __future__ import absolute_import

import logging

_LG = logging.getLogger(__name__)


class Agent(object):
    def __init__(self, env, agent_config, global_config):
        self.action_space = env.action_space
        self.observation_space = env.observation_space

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
        raise NotImplementedError('observe method is not implemented.')

    def act(self):
        """Choose action. Must be implemented in subclass."""
        raise NotImplementedError('act method is not implemented.')

    def reset(self, observation):
        """Reset agent with the initial state of the environment."""
        raise NotImplementedError('reset method is not implemented.')

    def run_post_episode_task(self):
        """Perform post episode task"""
        pass


class EpisodeRunner(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, timesteps):
        self.env = env
        self.agent = agent
        self.timesteps = timesteps

    def reset(self):
        obs = self.env.reset()
        self.agent.reset(obs)

    def run_post_episode_task(self):
        self.agent.run_post_episode_task()

    def start_monitor(self, outdir, **kwargs):
        self.env.monitor.start(outdir, **kwargs)

    def close_monitor(self):
        self.env.monitor.close()

    def run_episode(self, timesteps=None, render_mode=None):
        """Run one episode"""
        timesteps = timesteps or self.timesteps
        self.reset()

        if render_mode in ['human', 'static', 'random']:
            self.env.render(mode=render_mode)

        total_rewards = 0
        for t in range(timesteps):
            action = self.agent.act()
            observation, reward, done, info = self.env.step(action)
            self.agent.observe(action, observation, reward, done, info)

            total_rewards += reward
            if render_mode in ['human', 'static', 'random']:
                self.env.render(mode=render_mode)

            if done:
                t = t + 1
                break
        else:
            t = -1
        self.run_post_episode_task()
        return t, total_rewards
