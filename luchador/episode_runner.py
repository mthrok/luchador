from __future__ import absolute_import

import logging

_LG = logging.getLogger(__name__)


__all__ = ['EpisodeRunner']


class EpisodeRunner(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, max_timesteps=100):
        self.env = env
        self.agent = agent
        self.max_timesteps = max_timesteps
        self.n_episodes = 0

    def reset(self):
        obs = self.env.reset()
        self.agent.reset(obs)

    def perform_post_episode_task(self):
        self.agent.perform_post_episode_task()

    def run_episode(self, max_timesteps=None):
        """Run one episode"""
        max_timesteps = max_timesteps or self.max_timesteps
        self.reset()

        total_rewards = 0
        for t in range(max_timesteps):
            action = self.agent.act()
            observation, reward, done, info = self.env.step(action)
            self.agent.observe(action, observation, reward, done, info)
            total_rewards += reward

            if done:
                t = t + 1
                break
        else:
            t = -1
        self.perform_post_episode_task()
        return t, total_rewards
