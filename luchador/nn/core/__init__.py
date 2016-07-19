from __future__ import absolute_import

import luchador

if luchador.get_nn_backend() == 'tensorflow':
    from .tensorflow import *  # nopep8
else:
    from .theano import *  # nopep8
