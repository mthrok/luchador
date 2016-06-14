from __future__ import absolute_import


class LayerConfig(object):
    def __init__(self, layer, scope, input_tensor=None, output_tensor=None):
        self.layer = layer
        self.scope = scope
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor


class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        # Layer configurations
        self.layer_configs = []
        # I/O tensors of the model
        self.input_tensor = None
        self.output_tensor = None

    ###########################################################################
    def add(self, layer, scope=None):
        """Add layer to model

        Args:
          layer (Layer): Layer to add
          scope (str): Variable scope, used when instantiating layer.
        """
        self.layer_configs.append(LayerConfig(
            layer=layer,
            scope=scope,
            input_tensor=None,
            output_tensor=None,
        ))
        return self

    ###########################################################################
    # Functions for model-lavel copy and concatenation
    def __iadd__(self, other):
        """Append layers from another model after this model
        Args:
          other (Model): Model to be concatenated

        Returns:
          TFModel: Updated model
        """
        for cfg in other.layer_configs:
            self.add(layer=cfg.layer, scope=cfg.scope)
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
    # Functions for building actual computation graphs
    def __call__(self, input_tensor):
        return self.build(input_tensor)

    def build(self, input_tensor):
        raise NotImplementedError('`build` method is not implemented')

    ###########################################################################
    # Functions for retrieving variables and tensors
    def get_parameter_variables(self):
        """Get Variable objects consisting the parameters of this model"""
        ret = []
        for cfg in self.layer_configs:
            for variable in cfg.layer.parameter_variables.values():
                ret.append(variable)
        return ret

    def get_output_tensors(self):
        """Get Tensor objects which represent the output of each layer"""
        return [cfg.output_tensor for cfg in self.layer_configs]
