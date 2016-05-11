import logging

_LG = logging.getLogger(__name__)


class Agent(object):
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
            if not render_mode == 'noop':
                self.env.render(mode=render_mode)

            if done:
                return t+1, total_rewards
        return -1, total_rewards
