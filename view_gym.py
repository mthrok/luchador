import logging

import gym
from gym import envs
from gym import spaces

_LG = logging.getLogger(__name__)
_ENVS = sorted([env_spec for env_spec in envs.registry.all()])


def parse_env_name(arg):
    return _ENVS[int(arg)].id if arg.isdigit() else arg


def run(env):
    env.reset()
    rep = 0
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rep += 1
        if done or rep > 100:
            env.reset()
            rep = 0


def print_env_info(env):
    """Print environment info."""
    def print_space_summary(space):
        if isinstance(space, spaces.Box):
            _LG.debug('      Range:')
            for l, h in zip(space.low, space.high):
                _LG.debug('        [{}, {}]'.format(l, h))
    _LG.info('... Action Space: {}'.format(env.action_space))
    print_space_summary(env.action_space)
    _LG.info('... Observation Space: {}'.format(env.observation_space))
    print_space_summary(env.observation_space)
    return env


def init_logging(debug=False):
    _LG.setLevel(logging.DEBUG if debug else logging.INFO)


def help_str():
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


def parse_command_line_arguments():
    from argparse import RawTextHelpFormatter
    from argparse import ArgumentParser as AP
    ap = AP(
        formatter_class=RawTextHelpFormatter,
        description='Render examples of OpenAI/gym with random actions.',
    )
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('env', help=help_str())
    return ap.parse_args()


def main():
    args = parse_command_line_arguments()
    init_logging(args.debug)
    env = gym.make(parse_env_name(args.env))
    print_env_info(env)
    run(env)


if __name__ == '__main__':
    main()
