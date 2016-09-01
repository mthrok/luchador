from __future__ import absolute_import


import sys
import os
import logging

import numpy as np
from ale_python_interface import ALEInterface

from .base import Environment

_LG = logging.getLogger(__name__)

__all__ = ['ALEEnvironment']

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROM_DIR = os.path.join(_DIR, 'rom')


class ALEEnvironment(Environment):
    def __init__(
            self, rom,
            display_screen=False,
            sound=False,
            frame_skip=1,
            color_averaging=False,
            random_seed=0,
            random_start=None,
            record_screen_path=None,
            record_sound_filename=None,
            minimal_action_set=True,
    ):
        if display_screen and sys.platform == 'darwin':
            import pygame
            pygame.init()

        if not rom.endswith('.bin'):
            rom += '.bin'

        rom_path = os.path.join(_ROM_DIR, rom)
        if not os.path.isfile(rom_path):
            raise ValueError('ROM ({}) not found.'.format(self.rom))

        ale = ALEInterface()
        ale.setBool('sound', sound)
        ale.setBool('display_screen', display_screen)

        ale.setInt('frame_skip', frame_skip)
        ale.setBool('color_averaging', color_averaging)
        ale.setFloat('repeat_action_probability', 0.0)
        # Somehow this repeat_action_probability has unexpected effect on game.
        # The larger this value is, the more frames games take to restart.
        # And when 1.0 games completely hang
        # We are setting the default value of 0.0 here, expecting that
        # it has no effect as frame_skip == 1
        # This action repeating is agent's concern
        # so we do not implement an equivalent in our wrapper.

        ale.setInt('random_seed', random_seed)

        if record_screen_path:
            _LG.info('Recording screens to {}'.format(record_screen_path))
            if not os.path.exists(record_screen_path):
                os.makedirs(record_screen_path)
            ale.setString('record_screen_dir', record_screen_path)

        if record_sound_filename:
            _LG.info('Recording sound to {}'.format(record_sound_filename))
            record_sound_dir = os.path.dirname(record_sound_filename)
            if not os.path.exists(record_sound_dir):
                os.makedirs(record_sound_dir)
            ale.setBool('sound', True)
            ale.setString('record_sound_filename', record_sound_filename)

        if not rom.endswith('.bin'):
            rom += '.bin'

        ale.loadROM(rom_path)

        self.ale = ale
        self.rom = rom
        self.random_start = random_start
        self.minimal_action_set = minimal_action_set

        if self.minimal_action_set:
            self.actions = ale.getMinimalActionSet()
        else:
            self.actions = ale.getLegalActionSet()

    def __repr__(self):
        ale = self.ale
        return (
            '[Atari Env]\n'
            '    ROM: {}\n'
            '    display_screen           : {}\n'
            '    sound                    : {}\n'
            '    frame_skip               : {}\n'
            '    repeat_action_probability: {}\n'
            '    color_averaging          : {}\n'
            '    random_seed              : {}\n'
            '    random_start             : {}\n'
            '    record_screen_path       : {}\n'
            '    record_sound_filename    : {}\n'
            '    minimal_action_set       : {}\n'
            .format(
                self.rom,
                ale.getBool('display_screen'),
                ale.getBool('sound'),
                ale.getInt('frame_skip'),
                ale.getFloat('repeat_action_probability'),
                ale.getBool('color_averaging'),
                ale.getInt('random_seed'),
                self.random_start,
                ale.getString('record_screen_path'),
                ale.getString('record_sound_filename'),
                self.minimal_action_set,
            )
        )

    @property
    def n_actions(self):
        return len(self.actions)

    def reset(self):
        self.ale.reset_game()
        if self.random_start:
            for _ in range(1 + np.random.randint(self.random_start)):
                self.ale.act(0)
        return self._get_screen()

    def step(self, action):
        reward = self.ale.act(self.actions[action])
        screen = self._get_screen()
        terminal = self._is_terminal()
        info = {
            'lives': self.ale.lives(),
            'frame_number': self.ale.getFrameNumber(),
            'episode_frame_number': self.ale.getEpisodeFrameNumber(),
        }
        return reward, screen, terminal, info

    def _get_screen(self):
        return self.ale.getScreenRGB()

    def _is_terminal(self):
        return self.ale.game_over()
