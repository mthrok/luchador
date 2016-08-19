from __future__ import absolute_import

import inspect

import luchador.nn
import luchador.nn.models


def get_initializer(name):
    for name_, Class in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, luchador.nn.base.Initializer)
        ):
            return Class
    raise ValueError('Unknown Initializer: {}'.format(name))


def get_optimizer(name):
    for name_, Class in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, luchador.nn.base.Optimizer)
        ):
            return Class
    raise ValueError('Unknown Optimizer: {}'.format(name))


def get_model(name, **kwargs):
    for name_, func in inspect.getmembers(luchador.nn.models,
                                          inspect.isfunction):
        if name == name_:
            return func(**kwargs)
    raise ValueError('Unknown model name: {}'.format(name))
