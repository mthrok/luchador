from __future__ import print_function

from collections import defaultdict

import gym
import numpy as np

import luchador


class TabularQAgent(luchador.Agent):
    """TabularQAgent from gym example"""
    def __init__(self, env, agent_config={}, global_config=None):

        super(TabularQAgent, self).__init__(
            env, agent_config=agent_config, global_config=global_config
        )

        self.config = {
            'init_mean': 0.0,
            'init_std': 0.0,
            'learning_rate': 0.1,
            'eps': 0.05,
            'discount': 0.95,
            'n_iter': 10000,
        }
        self.config.update(agent_config)
        self.q = defaultdict(lambda: self._initial_q_value())

    def _initial_q_value(self):
        mean, std = self.config['init_mean'], self.config['init_std']
        return mean + std * np.random.randn(self.action_space.n)

    ###########################################################################
    # Functions for `reset` entry porint
    def reset(self, observation):
        self.observation_history = [observation]
        self.action_history = []
        self.reward_history = []
        self.done = False

    ###########################################################################
    # Functions for `observe` entry porint
    def _learn(self):
        """Update Q value based on the previous obs->act->rew->obs chain"""
        src = self.observation_history[-2]
        act = self.action_history[-1]
        rew = self.reward_history[-1]
        tgt = self.observation_history[-1]

        future = 0.0 if self.done else np.max(self.q[tgt])
        lr, beta = self.config['learning_rate'], self.config['discount']
        self.q[src][act] -= lr * (self.q[src][act] - rew - beta * future)

    def _update_history(self, action, observation, reward, done, info):
        self.action_history.append(action)
        self.observation_history.append(observation)
        self.reward_history.append(reward)
        self.done = done

    def observe(self, action, observation, reward, done, info):
        self._update_history(action, observation, reward, done, info)
        self._learn()

    ###########################################################################
    # Functions for `act` entry porint
    def _choose_action(self):
        """Choose an action with Epsilon-greedy policy"""
        if np.random.random() > self.config['eps']:
            last_obs = self.observation_history[-1]
            action = np.argmax(self.q[last_obs])
        else:
            action = self.action_space.sample()
        return action

    def act(self):
        return self._choose_action()


def main():
    env = gym.make('Roulette-v0')
    agt = TabularQAgent(env)
    runner = luchador.EpisodeRunner(env, agt, 100)

    runner.start_monitor('tmp')
    for i in range(100):
        print('Running episode {}'.format(i))
        t, r = runner.run_episode(timesteps=100, render_mode='noop')
        print('... {:16} Rewards: {}, Timesteps: {}'
              .format('NOT finished.' if t < 0 else 'Finished.', t, r))
    runner.close_monitor()

if __name__ == '__main__':
    main()
