from __future__ import absolute_import

import yaml
import inspect
import luchador

__all__ = ['sane_gym_import', 'load_config', 'get_env', 'get_agent']


def sane_gym_import():
    """Import Gym without polluting root logger"""
    import logging
    root_lg = logging.getLogger()
    root_handlers = root_lg.handlers
    root_lg.handlers = []

    import gym  # noqa: F401
    gym_lg = logging.getLogger('gym')
    gym_lg.handlers = root_lg.handlers
    gym_lg.propagate = False
    root_lg.handlers = root_handlers


def load_config(filepath):
    with open(filepath) as f:
        return yaml.load(f)


def get_agent(name):
    for name_, Class in inspect.getmembers(luchador, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, luchador.agent.base.Agent)
        ):
            return Class
    raise ValueError('Unknown Agent: {}'.format(name))


def get_env(name):
    for name_, Class in inspect.getmembers(luchador, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, luchador.env.base.Environment)
        ):
            return Class
    raise ValueError('Unknown Environment: {}'.format(name))
