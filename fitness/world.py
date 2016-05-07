import logging

_LG = logging.getLogger(__name__)


class World(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, timesteps):
        self.env = env
        self.agent = agent
        self.timesteps = timesteps

    def reset(self):
        self.agent.reset()
        return self.env.reset()

    def start_monitor(self, outdir, **kwargs):
        self.env.monitor.start(outdir, **kwargs)

    def close_monitor(self):
        self.env.monitor.close()

    def run_episode(self, timesteps=None, render_mode=True, initial_reward=-1):
        """Run one episode"""
        timesteps = self.timesteps if not timesteps else timesteps
        observation = self.reset()

        if not render_mode == 'noop':
            self.env.render(mode=render_mode)

        reward = initial_reward
        total_rewards = 0
        done = False
        for t in range(timesteps):
            action = self.agent.act(observation, reward, done)
            observation, reward, done, info = self.env.step(action)
            total_rewards += reward
            _LG.debug('... {}: {}, {}'.format(t, reward, observation))

            if not render_mode == 'noop':
                self.env.render(mode=render_mode)

            if done:
                return t+1, total_rewards
        return -1, total_rewards
