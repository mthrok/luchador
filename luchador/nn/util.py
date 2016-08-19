from __future__ import absolute_import

import inspect

import luchador.nn
import luchador.nn.models


def get_initializer(name):
    # TODO: Convert this to return instance
    for name_, class_ in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == name_ and
                issubclass(class_, luchador.nn.base.Initializer)
        ):
            return class_
    raise ValueError('Unknown Optimizer: {}'.format(name))


def get_optimizer(name):
    # TODO: Convert this to return instance
    for name_, class_ in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == name_ and
                issubclass(class_, luchador.nn.base.Optimizer)
        ):
            return class_
    raise ValueError('Unknown Optimizer: {}'.format(name))


def get_model(name, **kwargs):
    for name_, func in inspect.getmembers(luchador.nn.models,
                                          inspect.isfunction):
        if name == name_:
            return func(**kwargs)
    raise ValueError('Unknown model name: {}'.format(name))
