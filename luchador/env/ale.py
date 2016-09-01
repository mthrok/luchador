from __future__ import absolute_import


import sys
import os
import logging

import numpy as np
from ale_python_interface import ALEInterface

from .base import BaseEnvironment

_LG = logging.getLogger(__name__)

__all__ = ['ALEEnvironment']

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROM_DIR = os.path.join(_DIR, 'rom', 'atari')


class ALEEnvironment(BaseEnvironment):
    def __init__(
            self, rom,
            display_screen=False,
            sound=False,
            frame_skip=4,
            color_averaging=False,
            random_seed=0,
            random_start=None,
            record_screen_path=None,
            record_sound_filename=None,
            minimal_action_set=True,
            mode='train',
    ):
        if mode not in ['test', 'train']:
            raise ValueError('`mode` must be either `test` or `train`')

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

        ale.setInt('frame_skip', 1)  # Frame skip is implemented separately
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
        self.mode = mode
        self.frame_skip = frame_skip
        self.random_start = random_start
        self.minimal_action_set = minimal_action_set

        self.life_lost = False
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
            '    mode                     : {}\n'
            .format(
                self.rom,
                ale.getBool('display_screen'),
                ale.getBool('sound'),
                self.frame_skip,
                ale.getFloat('repeat_action_probability'),
                ale.getBool('color_averaging'),
                ale.getInt('random_seed'),
                self.random_start,
                ale.getString('record_screen_path'),
                ale.getString('record_sound_filename'),
                self.minimal_action_set,
                self.mode,
            )
        )

    @property
    def n_actions(self):
        return len(self.actions)

    def reset(self):
        self.life_lost = False
        self.ale.reset_game()
        if self.random_start:
            for _ in range(1 + np.random.randint(self.random_start)):
                self.ale.act(0)
        return self._get_screen()

    def step(self, action):
        reward = 0
        action = self.actions[action]

        self.life_lost = False
        initial_lives = self.ale.lives()
        for i in range(max(self.frame_skip, 1)):
            reward += self.ale.act(action)

            if not self.ale.lives() == initial_lives:
                self.life_lost = True

            terminal = self._is_terminal()
            if terminal:
                break
        screen = self._get_screen()
        info = {
            'lives': self.ale.lives(),
            'frame_number': self.ale.getFrameNumber(),
            'episode_frame_number': self.ale.getEpisodeFrameNumber(),
        }
        return reward, screen, terminal, info

    def _get_screen(self):
        return self.ale.getScreenRGB()

    def _is_terminal(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()
