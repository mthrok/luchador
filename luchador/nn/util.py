from __future__ import absolute_import

import inspect

import luchador.nn


def get_optimizer(name):
    for obj in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == obj[0] and
                issubclass(obj[1], luchador.nn.base.Optimizer)
        ):
            return obj[1]
    raise ValueError('Unknown Optimizer: {}'.format(name))
