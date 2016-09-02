from __future__ import absolute_import

import unittest

from luchador.env import ALEEnvironment as ALE


class ALEEnvironmentTest(unittest.TestCase):
    longMessage = True

    def test_rom_availability(self):
        """ROMs are available"""
        self.assertEqual(
            len(ALE.get_roms()), 61,
            'Not all the ALE ROMS are found. '
            'Run `python setup.py download_ale` from root directory '
            'to download ALE ROMs then re-install luchador.'
        )
