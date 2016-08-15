from __future__ import absolute_import

import inspect

import luchador.nn
import luchador.nn.models


def get_optimizer(name):
    for obj in inspect.getmembers(luchador.nn, inspect.isclass):
        if (
                name == obj[0] and
                issubclass(obj[1], luchador.nn.base.Optimizer)
        ):
            return obj[1]
    raise ValueError('Unknown Optimizer: {}'.format(name))


def get_model(name, **kwargs):
    for obj in inspect.getmembers(luchador.nn.models, inspect.isfunction):
        if name == obj[0]:
            model_maker_func = obj[1]
            return model_maker_func(**kwargs)
    raise ValueError('Unknown model name: {}'.format(name))
