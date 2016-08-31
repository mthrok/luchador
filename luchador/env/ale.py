from __future__ import absolute_import


import sys
import os
import logging

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
            repeat_action_probability=0.99,  # setting this to 1.0 makes
                                             # breakout hang
            color_averaging=False,
            random_seed=0,
            record_screen_path=None,
            record_sound_filename=None,
            minimal_action_set=True,
    ):
        if not rom.endswith('.bin'):
            rom += '.bin'

        self.rom = rom
        self.display_screen = display_screen
        self.sound = sound
        self.frame_skip = frame_skip
        self.repeat_action_probability = repeat_action_probability
        self.color_averaging = color_averaging
        self.random_seed = random_seed
        self.record_screen_path = record_screen_path
        self.record_sound_filename = record_sound_filename
        self.minimal_action_set = minimal_action_set

        self.ale = ALEInterface()
        self.init()

    def init(self):
        if self.display_screen and sys.platform == 'darwin':
            import pygame
            pygame.init()

        self.ale.setBool('display_screen', self.display_screen)
        self.ale.setBool('sound', self.sound)

        self.ale.setInt('frame_skip', self.frame_skip)
        self.ale.setBool('color_averaging', self.color_averaging)
        self.ale.setFloat('repeat_action_probability',
                          self.repeat_action_probability)

        if self.random_seed:
            self.ale.setInt('random_seed', self.random_seed)

        if self.record_screen_path:
            if not os.path.exists(self.record_screen_path):
                _LG.info('Creating folder %s' % self.record_screen_path)
                os.makedirs(self.record_screen_path)
            _LG.info('Recording screens to %s', self.record_screen_path)
            self.ale.setString('record_screen_dir', self.record_screen_path)

        if self.record_sound_filename:
            _LG.info('Recording sound to %s', self.record_sound_filename)
            self.ale.setBool('sound', True)
            self.ale.setString('record_sound_filename',
                               self.record_sound_filename)

        rom_path = os.path.join(_ROM_DIR, self.rom)
        if not os.path.isfile(rom_path):
            raise ValueError('ROM ({}) not found.'.format(self.rom))

        self.ale.loadROM(rom_path)

        if self.minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
        else:
            self.actions = self.ale.getLegalActionSet()

    def __repr__(self):
        return (
            '[Atari Env]\n'
            '    ROM: {}\n'
            '    display_screen           : {}\n'
            '    sound                    : {}\n'
            '    frame_skip               : {}\n'
            '    repeat_action_probability: {}\n'
            '    color_averaging          : {}\n'
            '    random_seed              : {}\n'
            '    record_screen_path       : {}\n'
            '    record_sound_filename    : {}\n'
            '    minimal_action_set       : {}\n'
            .format(
                self.rom, self.display_screen, self.sound, self.frame_skip,
                self.repeat_action_probability, self.color_averaging,
                self.random_seed, self.record_screen_path,
                self.record_sound_filename, self.minimal_action_set
            )
        )

    @property
    def n_actions(self):
        return len(self.actions)

    def reset(self):
        self.ale.reset_game()
        return self._get_screen()

    def step(self, action):
        reward = self.ale.act(self.actions[action])
        screen = self._get_screen()
        terminal = self._is_terminal()
        info = {'lives': self.ale.lives()}
        return reward, screen, terminal, info

    def _get_screen(self):
        return self.ale.getScreenRGB()

    def _is_terminal(self):
        return self.ale.game_over()
