from __future__ import absolute_import

import luchador

if luchador.get_nn_backend() == 'tensorflow':
    from .tensorflow import *  # noqa: F401, F403
else:
    from .theano import *  # noqa: F401, F403
