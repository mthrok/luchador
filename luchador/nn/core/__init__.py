from __future__ import absolute_import

import logging

import luchador


_LG = logging.getLogger(__name__)
_LG.info('Using {} backend'.format(luchador.get_nn_backend()))

if luchador.get_nn_backend() == 'tensorflow':
    from .tensorflow import *  # noqa: F401, F403
else:
    from .theano import *  # noqa: F401, F403
