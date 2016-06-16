from __future__ import absolute_import


def _sane_gym_import():
    """Import Gym while avoiding polluting root logger"""
    import logging
    root_lg = logging.getLogger()
    root_handlers = root_lg.handlers
    root_lg.handlers = []

    import gym  # nopep8
    gym_lg = logging.getLogger('gym')
    gym_lg.handlers = root_lg.handlers
    gym_lg.propagate = False
    root_lg.handlers = root_handlers

_sane_gym_import()

from .agent.base import Agent  # nopep8
from .episode_runner import EpisodeRunner  # nopep8
