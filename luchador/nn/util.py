from __future__ import absolute_import

import os
import inspect
import StringIO

import yaml

_CATALOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'catalog')


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


def get_model(name):
    from . import model
    for name_, Class in inspect.getmembers(model, inspect.isclass):
        if (
                name == name_ and
                issubclass(Class, model.Model)
        ):
            return Class
    raise ValueError('Unknown model: {}'.format(name))


def make_model(model_config):
    """Make model from model configuration

    Args:
      model_condig (JSON-compatible object): model configuration.

    Returns:
      Model
    """
    model = get_model(model_config['model_type'])()
    for cfg in model_config['layer_configs']:
        scope = cfg['scope']
        layer_cfg = cfg['layer']
        layer = get_layer(layer_cfg['name'])(**layer_cfg['args'])
        model.add_layer(layer=layer, scope=scope)
    return model


def get_model_config(model_name, **parameters):
    """Load pre-defined model configurations

    Args:
      model_name (name): Model name
      parameters: Parameter for model config

    Returns:
      JSON-compatible object: model configuration.
    """
    file_name = '{}.yml'.format(model_name)
    if os.path.isfile(file_name):
        file_path = file_name
    elif os.path.isfile(os.path.join(_CATALOG_DIR, file_name)):
        file_path = os.path.join(_CATALOG_DIR, file_name)
    else:
        raise ValueError(
            'No model definition file ({}) found.'.format(file_name))

    with open(file_path, 'r') as f:
        model_text = f.read()

    if parameters:
        model_text = model_text.format(**parameters)

    model_text = StringIO.StringIO(model_text)
    return yaml.load(model_text)
