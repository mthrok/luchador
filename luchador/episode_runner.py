from __future__ import absolute_import

import logging

_LG = logging.getLogger(__name__)


__all__ = ['GymEpisodeRunner']


class GymEpisodeRunner(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, timesteps):
        self.env = env
        self.agent = agent
        self.timesteps = timesteps

    def reset(self):
        obs = self.env.reset()
        self.agent.reset(obs)

    def perform_post_episode_task(self):
        self.agent.perform_post_episode_task()

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
        self.perform_post_episode_task()
        return t, total_rewards
