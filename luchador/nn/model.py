from __future__ import absolute_import

from collections import OrderedDict

from .core import scope as scp

__all__ = ['Model']


class LayerConfig(object):
    def __init__(self, layer, scope, input=None, output=None):
        self.layer = layer
        self.scope = scope
        self.input = input
        self.output = output

    def __eq__(self, other):
        if isinstance(other, LayerConfig):
            return self.layer == other.layer and self.scope == other.scope
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '{{scope: {}, layer: {}, input: {}, output: {}}}'.format(
            self.scope, self.layer, self.input, self.output)


class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        # Layer configurations
        self.layer_configs = []
        # I/O tensors of the model
        self.input = None
        self.output = None

    ###########################################################################
    def add_layer(self, layer, scope=None):
        """Add layer to model

        Args:
          layer (Layer): Layer to add
          scope (str): Variable scope, used when instantiating layer.
        """
        self.layer_configs.append(LayerConfig(
            layer=layer,
            scope=scope,
            input=None,
            output=None,
        ))
        return self

    ###########################################################################
    # Functions for model-wise copy and concatenation
    def __iadd__(self, other):
        """Append layers from another model after this model
        Args:
          other (Model): Model to be concatenated

        Returns:
          TFModel: Updated model
        """
        for cfg in other.layer_configs:
            self.add_layer(layer=cfg.layer, scope=cfg.scope)
        return self

    def copy(self):
        """Create new model instance with the same configuration"""
        new_model = type(self)()
        new_model += self
        return new_model

    def __add__(self, other):
        """Create new model which contains layers from other model after this model
        Args:
          other (Model): Model to be concatenated

        Returns:
          Model: Resulting new model
        """
        new_model = self.copy()
        new_model += other
        return new_model

    ###########################################################################
    # Model definition equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.layer_configs == other.layer_configs
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    ###########################################################################
    # Functions for building actual computation graphs
    def __call__(self, input):
        return self.build(input)

    def build(self, input):
        """Build the model on top of input tensor"""
        tensor = self.input = input()
        for cfg in self.layer_configs:
            cfg.input = tensor
            with scp.variable_scope(cfg.scope or scp.get_variable_scope()):
                tensor = cfg.layer(tensor)
            cfg.output = tensor
        self.output = tensor
        return self.output

    ###########################################################################
    # Functions for retrieving variables and tensors
    def get_parameter_variables(self):
        """Get Variable objects consisting the parameters of this model"""
        ret = OrderedDict()
        base_name = scp.get_variable_scope().name
        if base_name:
            base_name = '{}/'.format(base_name)
        for cfg in self.layer_configs:
            scope = cfg.scope
            layer_name = scope if isinstance(scope, str) else scope.name
            for var_name, var in cfg.layer.parameter_variables.items():
                name = '{}{}/{}'.format(base_name, layer_name, var_name)
                ret[name] = var
        return ret

    def get_output_tensors(self):
        """Get Tensor objects which represent the output of each layer"""
        ret = OrderedDict()
        base_name = scp.get_variable_scope().name
        if base_name:
            base_name = '{}/'.format(base_name)
        for cfg in self.layer_configs:
            scope = cfg.scope
            layer_name = scope if isinstance(scope, str) else scope.name
            name = '{}{}/{}'.format(base_name, layer_name, cfg.output.name)
            ret[name] = cfg.output
        return ret

    ###########################################################################
    def serialize(self):
        """Serialize model configuration"""
        return {
            'layer_configs': [{
                'scope': cfg.scope,
                'layer': cfg.layer.serialize(),
            } for cfg in self.layer_configs]
        }

    ###########################################################################
    def __repr__(self):
        return str(self.layer_configs)
