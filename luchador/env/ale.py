from __future__ import absolute_import


import sys
import os
import logging

import numpy as np
from scipy.misc import imresize
from ale_python_interface import ALEInterface

from .base import BaseEnvironment

_LG = logging.getLogger(__name__)

__all__ = ['ALEEnvironment']

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROM_DIR = os.path.join(_DIR, 'rom', 'atari')


class ALEEnvironment(BaseEnvironment):
    def __init__(
            self, rom,
            mode='train',
            width=84,
            height=84,
            grayscale=True,
            frame_skip=4,
            minimal_action_set=True,
            random_seed=0,
            random_start=None,
            record_screen_path=None,
            record_sound_filename=None,
            display_screen=False,
            sound=False,
    ):
        # TODO: Add individual unittest
        if mode not in ['test', 'train']:
            raise ValueError('`mode` must be either `test` or `train`')

        if not rom.endswith('.bin'):
            rom += '.bin'

        rom_path = os.path.join(_ROM_DIR, rom)
        if not os.path.isfile(rom_path):
            raise ValueError('ROM ({}) not found.'.format(self.rom))

        if display_screen and sys.platform == 'darwin':
            import pygame
            pygame.init()

        ale = ALEInterface()
        ale.setBool('sound', sound)
        ale.setBool('display_screen', display_screen)

        # Frame skip is implemented separately
        ale.setInt('frame_skip', 1)
        ale.setBool('color_averaging', False)
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
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.frame_skip = frame_skip
        self.random_start = random_start
        self.minimal_action_set = minimal_action_set

        self.life_lost = False
        if minimal_action_set:
            self.actions = ale.getMinimalActionSet()
        else:
            self.actions = ale.getLegalActionSet()

        if height or width:
            orig_width, orig_height = self.ale.getScreenDims()
            height = height or orig_height
            width = width or orig_width
            self.size = (height, width) if grayscale else (height, width, 3)
        else:
            self.size = None

        if grayscale:
            self._get_screen = self._get_screen_grayscale
        else:
            self._get_screen = self._get_screen_rgb

    def __repr__(self):
        ale = self.ale
        return (
            '[Atari Env]\n'
            '    ROM: {}\n'
            '    display_screen           : {}\n'
            '    sound                    : {}\n'
            '    grayscale                : {}\n'
            '    frame_skip               : {}\n'
            '    random_seed              : {}\n'
            '    random_start             : {}\n'
            '    record_screen_path       : {}\n'
            '    record_sound_filename    : {}\n'
            '    minimal_action_set       : {}\n'
            '    mode                     : {}\n'
            '    n_actions                : {}\n'
            .format(
                self.rom,
                ale.getBool('display_screen'),
                ale.getBool('sound'),
                self.grayscale,
                self.frame_skip,
                ale.getInt('random_seed'),
                self.random_start,
                ale.getString('record_screen_path'),
                ale.getString('record_sound_filename'),
                self.minimal_action_set,
                self.mode,
                self.n_actions,
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
        else:
            self.ale.act(0)

        return self._get_observation()

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
        observation = self._get_observation()
        info = {
            'lives': self.ale.lives(),
            'total_frame_number': self.ale.getFrameNumber(),
            'episode_frame_number': self.ale.getEpisodeFrameNumber(),
        }
        return reward, observation, terminal, info

    def _get_screen_grayscale(self):
        return self.ale.getScreenGrayscale()[:, :, 0]

    def _get_screen_rgb(self):
        return self.ale.getScreenRGB()

    def _get_observation(self):
        screen = self._get_screen()
        if self.size:
            return imresize(screen, self.size)
        return screen

    def _is_terminal(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()
