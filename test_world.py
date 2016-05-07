import sys
import logging
import inspect

import gym
from gym import envs
from gym import spaces

from fitness import agent
from fitness import world

_LG = logging.getLogger('fitness')
_ENVS = sorted([env_spec for env_spec in envs.registry.all()])
_AGENTS = sorted([obj[1] for obj in inspect.getmembers(
    sys.modules['fitness.agent'], inspect.isclass
    ) if issubclass(obj[1], agent._Agent) and not obj[1] == agent._Agent])


def print_env_info(env):
    """Print environment info."""
    def print_space_summary(space):
        if isinstance(space, spaces.Tuple):
            for sp in spaces:
                print_space_summary(sp)
        if isinstance(space, spaces.Discrete):
            _LG.info('      Range: [0, {}]'.format(space.n-1))
        if isinstance(space, spaces.Box):
            _LG.info('      Range: [{}, {}]'.format(space.low, space.high))
    _LG.info('... Action Space: {}'.format(env.action_space))
    print_space_summary(env.action_space)
    _LG.info('... Observation Space: {}'.format(env.observation_space))
    print_space_summary(env.observation_space)
    return env


def create_env(arg):
    env_entry = _ENVS[int(arg)].id if arg.isdigit() else arg
    return gym.make(env_entry)


def create_agent(arg, env):
    agent_class = _AGENTS[int(arg)] if arg.isdigit() else arg
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
        ret += '{:>6d}: {}\n'.format(i, agt.__name__)
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
    ap.add_argument('--outdir',
                    help=('Output directory for monitoring.\n'
                          'Unless given, monitoring is disabled.'))
    ap.add_argument('--timesteps', type=int, default=100)
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--render-mode', default='human',
                    choices=['noop', 'random', 'static', 'human'])
    return ap.parse_args()


def init_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    _LG.setLevel(level)
    logging.getLogger('fitness').setLevel(level)


def main():
    args = parse_command_line_arguments()
    init_logging(args.debug)
    env = create_env(args.env)
    agt = create_agent(args.agent, env)
    wrd = world.World(env, agt, 100)
    print_env_info(env)
    if args.outdir:
        wrd.start_monitor(args.outdir)

    for i in range(args.episodes):
        _LG.info('Running an episode {}'.format(i))
        t, r = wrd.run_episode(
            timesteps=args.timesteps, render_mode=args.render_mode)
        if t < 0:
            _LG.info('Did not finish')
        else:
            _LG.info('Finished with {} steps. Rewards: {}'.format(t, r))

    wrd.close_monitor()

if __name__ == '__main__':
    main()
