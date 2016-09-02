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
    @staticmethod
    def get_roms():
        """Get the list of ROMs available

        Returns:
          list of srting: Names of available ROMs
        """
        return [rom for rom in os.listdir(_ROM_DIR)
                if rom.endswith('.bin')]

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
            buffer_frames=2,
            preprocess_mode='max',
            display_screen=False,
            play_sound=False,
            record_screen_path=None,
            record_sound_filename=None,
    ):
        """Initialize ALE Environment with the given parmeters

        Args:
          rom (str): ROM name. Use `get_roms` for the list of available ROMs.

          mode (str): When `train`, a loss of life is considered as terminal
                      condition. When `test`, a loss of life is not considered
                      as terminal condition.

          width (int or None): Output screen width.
                               If None the original width is used.

          height (int or None): Output screen height.
                                If None the original height is used.

          grayscale(bool): If True, output screen is gray scale and
                           has no color channel. i.e. output shape == (h, w)
                           Other wise output screen has color channel with
                           shape (h, w, 3)

          frame_skip(int): When calling `step` method, action is repeated for
                           this numebr of times, internally, unless a terminal
                           condition is met.

          minimal_action_set(bool):
              When True, `n_actions` property reports actions only meaningfull
              to the loaded ROM. Otherwise all the 18 actions are dounted.

          random_seed(int): ALE's random seed

          random_start(int or None): When given, at most this number of frames
              are played with action == 0. This technique is often used to
              prevent environment from transitting deterministically, but
              in case of ALE, as reset command does not reset system state
              we may not need to do this. (TODO: Check.)

          buffer_frames(int): The number of latest frame to preprocess.

          preprocess_mode(str): Either `max` or `average`. When obtaining
                                observation, pixel-wise maximum or average
                                over buffered frames are taken before resizing

          display_screen(bool): Display sceen when True.

          play_sound(bool): Play sound

          record_screen_path(str): Passed to ALE. Save the original screens
              into the path.
              Note: that this is different from the observation returned by
              `step` method.

          record_screen_filename(str): Passed to ALE. Save sound to a file.

        """
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

        self.rom = rom
        self.mode = mode
        self.life_lost = False

        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.frame_skip = frame_skip
        self.random_start = random_start
        self.minimal_action_set = minimal_action_set

        self._init_ale(rom_path=rom_path,
                       random_seed=random_seed, random_start=random_start,
                       display_screen=display_screen, play_sound=play_sound,
                       record_screen_path=record_screen_path,
                       record_sound_filename=record_sound_filename)



        if minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
        else:
            self.actions = self.ale.getLegalActionSet()

        orig_width, orig_height = self.ale.getScreenDims()
        height = height or orig_height
        width = width or orig_width
        if height == orig_height and width == orig_width:
            self.resize = None
        else:
            self.resize = (height, width) if grayscale else (height, width, 3)

        if grayscale:
            self._get_screen = self._get_screen_grayscale
        else:
            self._get_screen = self._get_screen_rgb

    def _init_ale(
            self, rom_path, random_seed, random_start, display_screen,
            play_sound, record_screen_path, record_sound_filename):
        ale = ALEInterface()
        ale.setBool('sound', play_sound)
        ale.setBool('display_screen', display_screen)
        ale.setInt('random_seed', random_seed)

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

        ale.loadROM(rom_path)

        self.ale = ale

    def __repr__(self):
        ale = self.ale
        return (
            '[Atari Env]\n'
            '    ROM: {}\n'
            '    display_screen           : {}\n'
            '    sound                    : {}\n'
            '    resize                   : {}\n'
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
                self.resize,
                self.grayscale,
                self.frame_skip,
                ale.getInt('random_seed'),
                self.random_start,
                ale.getString('record_screen_path') or None,
                ale.getString('record_sound_filename') or None,
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

        rand = self.random_start
        repeat = 1 + (np.random.randint(rand) if rand else 0)
        for _ in range(repeat):
            self._step(0)
        return self._get_observation()

    def step(self, action):
        reward = 0
        action = self.actions[action]

        self.life_lost = False
        initial_lives = self.ale.lives()
        for i in range(max(self.frame_skip, 1)):
            reward += self._step(action)

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

    def _step(self, action):
        reward = self.ale.act(action)
        return reward

    def _get_screen_grayscale(self):
        return self.ale.getScreenGrayscale()[:, :, 0]

    def _get_screen_rgb(self):
        return self.ale.getScreenRGB()

    def _get_observation(self):
        screen = self._get_screen()
        if self.resize:
            return imresize(screen, self.resize)
        return screen

    def _is_terminal(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()
