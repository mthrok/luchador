from __future__ import absolute_import


import sys
import os
import logging

from ale_python_interface import ALEInterface

_LG = logging.getLogger(__name__)

__all__ = ['ALEEnvironment']

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROM_DIR = os.path.join(_DIR, 'rom')


class ALEEnvironment(object):
    def __init__(self, rom_name, **kwargs):
        if not rom_name.endswith('.bin'):
            rom_name += '.bin'

        rom_path = os.path.join(_ROM_DIR, rom_name)
        if not os.path.isfile(rom_path):
            raise ValueError('ROM ({}) not found.'.format(rom_name))

        self.ale = ALEInterface()
        self._parse_args(rom_path, **kwargs)

    def _parse_args(
            self,
            rom_file,
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
        if display_screen and sys.platform == 'darwin':
            import pygame
            pygame.init()

        self.ale.setBool('display_screen', display_screen)
        self.ale.setBool('sound', sound)

        self.ale.setInt('frame_skip', frame_skip)
        self.ale.setBool('color_averaging', color_averaging)
        self.ale.setFloat('repeat_action_probability',
                          repeat_action_probability)

        if random_seed:
            self.ale.setInt('random_seed', random_seed)

        if record_screen_path:
            if not os.path.exists(record_screen_path):
                _LG.info('Creating folder %s' % record_screen_path)
                os.makedirs(record_screen_path)
            _LG.info('Recording screens to %s', record_screen_path)
            self.ale.setString('record_screen_dir', record_screen_path)

        if record_sound_filename:
            _LG.info('Recording sound to %s', record_sound_filename)
            self.ale.setBool('sound', True)
            self.ale.setString('record_sound_filename', record_sound_filename)

        self.ale.loadROM(rom_file)

        if minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
        else:
            self.actions = self.ale.getLegalActionSet()

        _LG.info(
            '\n'
            '[Atari Env]'
            '    ROM: {}'
            '    display_screen           : {}'
            '    sound                    : {}'
            '    frame_skip               : {}'
            '    repeat_action_probability: {}'
            '    color_averaging          : {}'
            '    random_seed              : {}'
            '    record_screen_path       : {}'
            '    record_sound_filename    : {}'
            '    minimal_action_set       : {}'
            .format(
                rom_file, display_screen, sound, frame_skip,
                repeat_action_probability, color_averaging,
                random_seed, record_screen_path, record_sound_filename,
                minimal_action_set
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
