import os
import sys
import json
import time
import logging
import inspect
import datetime

import yaml
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


###############################################################################
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


def create_agent(agent_name, agent_config, env, global_config):
    _LG.info('Making new agent: {}'.format(agent_name))
    agent_class = getattr(agent, agent_name)
    return agent_class(
        env, agent_config=agent_config, global_config=global_config)


def init_logging(debug=False):
    _LG.setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger('gym').setLevel(logging.INFO)


def main(config):
    init_logging(config['debug'])
    _LG.info('Params: \n{}'.format(
        json.dumps(config, indent=2, sort_keys=True)))

    env = gym.make(config['env'])
    agt = create_agent(config['agent'], config['agent_config'], env, config)
    runner = luchador.EpisodeRunner(env, agt, 100)
    print_env_info(env)

    monitor = config['monitor']
    if monitor['enable']:
        runner.start_monitor(monitor['output_dir'], force=monitor['force'])

    exercise = config['exercise']
    for i in range(exercise['episodes'] + 1):
        _LG.info('Running episode {}'.format(i))
        t0 = time.time()
        t, r = runner.run_episode(
            timesteps=exercise['timesteps'],
            render_mode=monitor['render_mode'])
        dt = time.time() - t0
        _LG.info('... {}, Rewards: {}, Timesteps: {} ({} [sec])'
                 .format('NOT finished' if t < 0 else 'Finished', r, t, dt))
    runner.close_monitor()


###############################################################################
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


def _parse_env_name(arg):
    return _ENVS[int(arg)].id if arg.isdigit() else arg


def _parse_agent_name(arg):
    return _AGENTS[int(arg)] if arg.isdigit() else arg


def _parse_command_line_arguments():
    default_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'exercise_config.yml'
    )
    import argparse
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Run environment with agents.',
    )
    ap.add_argument('env', help=_env_help_str())
    ap.add_argument('agent', help=_agent_help_str())
    ap.add_argument('--episodes', '-ep', type=int)
    ap.add_argument('--timesteps', '-ts', type=int)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument(
        '--force', action='store_true',
        help='Overwrite monitoring data.'
    )
    ap.add_argument(
        '--config', '-c', type=argparse.FileType('r'),
        default=open(default_config, 'r'),
        help='YAML file containing the configuration.'
    )
    ap.add_argument(
        '--output-dir', '-o',
        help=('Base output directory for monitoring.\n'
              'Actual monitoring data is stored in \n'
              'subdirectory with runtime unique name.\n'
              'Default: "monitoring"')
    )
    ap.add_argument(
        '--no-monitor', action='store_true', help='Disable monitoring')
    ap.add_argument(
        '--render-mode', choices=['noop', 'random', 'static', 'human'])
    return ap.parse_args()


def _get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def _parse_config():
    args = _parse_command_line_arguments()
    config = yaml.load(args.config)
    config['env'] = _parse_env_name(args.env)
    config['agent'] = _parse_agent_name(args.agent)
    if args.debug:
        config['debug'] = True
    if args.episodes:
        config['exercise']['episodes'] = args.episodes
    if args.timesteps:
        config['exercise']['timesteps'] = args.timesteps
    if args.force:
        config['monitor']['force'] = True
    if args.no_monitor:
        config['monitor']['enable'] = False
    if args.output_dir:
        config['monitor']['output_dir'] = args.output_dir
    if args.render_mode:
        config['monitor']['render_mode'] = args.render_mode

    monitor = config['monitor']
    monitor['output_dir'] = os.path.join(
        monitor['output_dir'], '{}_{}_{}'.format(
            config['env'], config['agent'], _get_current_time()))
    return config


def entry_point():
    main(_parse_config())
