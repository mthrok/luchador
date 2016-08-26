from __future__ import absolute_import

import inspect


def get_initializer(name):
    import luchador.nn
    for name_, Class in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, luchador.nn.core.base.Initializer)
        ):
            return Class
    raise ValueError('Unknown Initializer: {}'.format(name))


def get_optimizer(name):
    import luchador.nn
    for name_, Class in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, luchador.nn.core.base.Optimizer)
        ):
            return Class
    raise ValueError('Unknown Optimizer: {}'.format(name))


def get_layer(name):
    import luchador.nn
    for name_, Class in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, luchador.nn.core.base.layer.Layer)
        ):
            return Class
    raise ValueError('Unknown Layer: {}'.format(name))


def get_model(name, **kwargs):
    from . import models
    for name_, func in inspect.getmembers(models, inspect.isfunction):
        if name == name_:
            return func(**kwargs)
    raise ValueError('Unknown model name: {}'.format(name))
