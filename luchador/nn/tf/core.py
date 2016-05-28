import tensorflow as tf

from ..core import Model as BaseModel
from ..core import Layer as BaseLayer


class TFModel(BaseModel):
    """Add interface for concatenation and summarization to BaseLayer"""
    def __init__(self):
        super(TFModel, self).__init__()

        self.output_summary_ops = None
        self.parameter_summary_ops = None

    def __call__(self, input_tensor):
        """Build the model on top on input_tensor"""
        self.layer_outputs = []
        self.layer_parameters = {}
        self.input_tensor = input_tensor
        for layer in self.layers:
            input_tensor = layer(input_tensor)
            self.layer_outputs.append(input_tensor)
            for name, var in layer.parameter_variables.items():
                key = '{}/{}'.format(layer.args['scope'] or '', name)
                self.layer_parameters[key] = var
        self.output_tensor = input_tensor
        return self.output_tensor

    def copy(self):
        """Create a new TFModel instance with the same configuration"""
        new_model = TFModel()
        new_model += self
        return new_model

    def __iadd__(self, other):
        """Append layers from other model after this model
        Args:
          other (TFModel): TFModel to be concatenated

        Returns:
          TFModel: Updated model
        """
        for layer in other.layers:
            self.add(layer.copy())
        return self

    def __add__(self, other):
        """Concatenate other model after this model
        Args:
          other (TFModel): TFModel to be concatenated

        Returns:
          TFModel: Resulting new model
        """
        new_model = self.copy()
        new_model += other
        return new_model

    ###########################################################################
    def _summarize(self, tensor):
        return tf.histogram_summary(tensor.name, tensor)

    def get_output_summary_ops(self):
        if self.output_summary_ops is None:
            self.output_summary_ops = [
                self._summarize(layer_output)
                for layer_output in self.layer_outputs
            ]
        return self.output_summary_ops

    def get_parameter_summary_ops(self):
        if self.parameter_summary_ops is None:
            self.parameter_summary_ops = [
                self._summarize(param)
                for param in self.layer_parameters.values()
            ]
        return self.parameter_summary_ops


class TFLayer(BaseLayer):
    """Add copying and scoping interface to BaseLayer"""
    def __init__(self, args):
        """Initialize attributes required for common TFLayer operations

        Args:
          args (dict): Arguments used to initialize the subclass instance.
            This will be used to recreate a subclass instance.
        """
        super(TFLayer, self).__init__()

        self.args = args
        self.scope = args.get('scope')

        self.parameter_variables = {}

    ###########################################################################
    def get_scope(self):
        if self.scope is None:
            return tf.get_variable_scope()
        return self.scope

    ###########################################################################
    def copy(self):
        """Copy layer configuration"""
        return type(self)(**self.args)
