import os
import sys
import logging
import inspect
import datetime

import gym
from gym import envs
from gym import spaces

import luchador
from luchador import agent

_LG = logging.getLogger('luchador')
_ENVS = sorted([env_spec for env_spec in envs.registry.all()])
_AGENTS = sorted([obj[1].__name__ for obj in inspect.getmembers(
    sys.modules['luchador.agent'], inspect.isclass
    ) if issubclass(obj[1], luchador.core.Agent)])


def print_env_info(env):
    """Print environment info."""
    def print_space_summary(space):
        if isinstance(space, spaces.Tuple):
            for sp in space.spaces:
                print_space_summary(sp)
        if isinstance(space, spaces.Discrete):
            _LG.info('      Range: [0, {}]'.format(space.n-1))
    _LG.info('... Action Space: {}'.format(env.action_space))
    print_space_summary(env.action_space)
    _LG.info('... Observation Space: {}'.format(env.observation_space))
    print_space_summary(env.observation_space)
    _LG.info('... Reward Range: {}'.format(env.reward_range))
    return env


def parse_env_name(arg):
    return _ENVS[int(arg)].id if arg.isdigit() else arg


def parse_agent_name(arg):
    return _AGENTS[int(arg)] if arg.isdigit() else arg


def create_agent(agent_name, env):
    _LG.info('Making new egent: {}'.format(agent_name))
    agent_class = getattr(agent, agent_name)
    return agent_class(action_space=env.action_space,
                       observation_space=env.observation_space)


def _env_help_str():
    ret = (
        'The followings are the environments installed in this machine.\n'
        'You can choose environment wither by its name or index.\n'
        'Note: Some environments requrie additional dependencies to be'
        ' installed.\n\n'
    )
    for i, env_spec in enumerate(_ENVS):
        ret += '{:>6d}: {} ({})\n'.format(
            i, env_spec.id, env_spec._entry_point)
    return ret + '\n'


def _agent_help_str():
    ret = (
        'The followings are the agents available.\n'
        'You can choose environment wither by its name or index.\n'
        'Note: Not all the combination of Env/Agent is possible.\n\n'
    )
    for i, agt in enumerate(_AGENTS):
        ret += '{:>6d}: {}\n'.format(i, agt)
    return ret + '\n'


def parse_command_line_arguments():
    from argparse import RawTextHelpFormatter
    from argparse import ArgumentParser as AP
    ap = AP(
        formatter_class=RawTextHelpFormatter,
        description='Run environment with agents.',
    )
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('env', help=_env_help_str())
    ap.add_argument('agent', help=_agent_help_str())
    ap.add_argument('--outdir', '-o', default='monitoring',
                    help=('Base output directory for monitoring.\n'
                          'Actual monitoring data is stored in \n'
                          'subdirectory with runtime unique name.'))
    ap.add_argument('--no-monitor', action='store_true',
                    help='Disable monitoring')
    ap.add_argument('--timesteps', '-ts', type=int, default=1000)
    ap.add_argument('--episodes', '-ep', type=int, default=200)
    ap.add_argument('--render-mode', default='human',
                    choices=['noop', 'random', 'static', 'human'])
    return ap.parse_args()


def init_logging(debug=False):
    _LG.setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger('gym').setLevel(logging.INFO)


def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def main():
    args = parse_command_line_arguments()
    init_logging(args.debug)
    env_name = parse_env_name(args.env)
    agent_name = parse_agent_name(args.agent)

    env = gym.make(env_name)
    agt = create_agent(agent_name, env)
    wrd = luchador.World(env, agt, 100)
    print_env_info(env)

    if not args.no_monitor:
        time_str = get_current_time()
        outdir = os.path.join(
            args.outdir, '{}_{}_{}'.format(env_name, agent_name, time_str))
        wrd.start_monitor(outdir)

    for i in range(args.episodes):
        _LG.info('Running episode {}'.format(i))
        t, r = wrd.run_episode(
            timesteps=args.timesteps, render_mode=args.render_mode)
        if t < 0:
            _LG.info('... Did not finish')
        else:
            _LG.info('... Finished with {} steps. Rewards: {}'.format(t, r))

    wrd.close_monitor()
