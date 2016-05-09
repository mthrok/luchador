import logging

_LG = logging.getLogger(__name__)


class World(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, timesteps):
        self.env = env
        self.agent = agent
        self.timesteps = timesteps

    def reset(self):
        obs = self.env.reset()
        self.agent.reset(obs)

    def start_monitor(self, outdir, **kwargs):
        self.env.monitor.start(outdir, **kwargs)

    def close_monitor(self):
        self.env.monitor.close()

    def run_episode(self, timesteps=None, render_mode=True):
        """Run one episode"""
        timesteps = self.timesteps if not timesteps else timesteps
        self.reset()

        if not render_mode == 'noop':
            self.env.render(mode=render_mode)

        total_rewards = 0
        for t in range(timesteps):
            action = self.agent.act()
            observation, reward, done, info = self.env.step(action)
            self.agent.observe(action, observation, reward, done, info)

            total_rewards += reward
            _LG.debug('... {}: {}, {}'.format(t, reward, observation))

            if not render_mode == 'noop':
                self.env.render(mode=render_mode)

            if done:
                return t+1, total_rewards
        return -1, total_rewards
