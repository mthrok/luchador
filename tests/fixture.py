from __future__ import absolute_import

import inspect
import numpy as np

import luchador


def create_image(height=210, width=160, channel=3):
    if channel:
        shape = (height, width, channel)
    else:
        shape = (height, width)
    return np.ones(shape, dtype=np.uint8)


def get_initializers():
    return {
        name: Class for name, Class
        in inspect.getmembers(luchador.nn, inspect.isclass)
        if issubclass(Class, luchador.nn.base.Initializer)
    }
