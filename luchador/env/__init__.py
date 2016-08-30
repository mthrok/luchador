from __future__ import absolute_import

import logging
_LG = logging.getLogger(__name__)

try:
    from .ale import *  # noqa: F401, F403
except ImportError:
    _LG.exception('Failed to import ALE Environment.')
