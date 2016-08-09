from __future__ import absolute_import

__all__ = ['sane_gym_import']


def sane_gym_import():
    """Import Gym without polluting root logger"""
    import logging
    root_lg = logging.getLogger()
    root_handlers = root_lg.handlers
    root_lg.handlers = []

    import gym  # nopep8
    gym_lg = logging.getLogger('gym')
    gym_lg.handlers = root_lg.handlers
    gym_lg.propagate = False
    root_lg.handlers = root_handlers
