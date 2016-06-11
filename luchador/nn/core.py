from __future__ import absolute_import


class BaseLayer(object):
    """Class for holding layer configuratiions"""
    def __init__(self, args):
        """Initialize attributes required for common operations

        Args:
          args (dict): Arguments used to initialize the subclass instance.
            This will be used to recreate a subclass instance.
        """
        super(BaseLayer, self).__init__()

        self.args = args
        self.parameter_variables = {}

    def copy(self):
        """Create new layer instance with the same configuration"""
        return type(self)(**self.args)

    def __call__(self, input_tensor):
        """Build layer on top of the given tensor"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        raise NotImplementedError(
            '`build` method is not implemented for {} class.'
            .format(type(self).__name__)
        )


class BaseModel(object):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Layer configurations
        self.layer_configs = []
        # I/O tensors of the model
        self.input_tensor = None
        self.output_tensor = None

    def add(self, layer, scope=None):
        """Add layer to model

        Args:
          layer (Layer): Layer to add
          scope (str): Variable scope, used when instantiating layer.
        """
        self.layer_configs.append({
            'layer': layer,
            'scope': scope,
            'input_tensor': None,
            'output_tensor': None,
        })
        return self

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

    def __iadd__(self, other):
        """Append layers from another model after this model
        Args:
          other (Model): Model to be concatenated

        Returns:
          TFModel: Updated model
        """
        for layer_config in other.layer_configs:
            layer, scope = layer_config['layer'], layer_config['scope']
            self.add(layer=layer, scope=scope)
        return self

    def copy(self):
        """Create new model instance with the same configuration"""
        new_model = type(self)()
        new_model += self
        return new_model

    def __call__(self, input_tensor):
        return self.build(input_tensor)

    def build(self, input_tensor):
        raise NotImplementedError(
            '`build` method is not implemented for class: {}.'
            .format(type(self).__name__)
        )
