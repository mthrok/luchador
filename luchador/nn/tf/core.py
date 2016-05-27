import tensorflow as tf

from ..core import Model as BaseModel
from ..core import Layer as BaseLayer


class TFModel(BaseModel):
    def __init__(self, input_shape=None, output_shape=None):
        super(TFModel, self).__init__()
        self.args = {'input_shape': input_shape, 'output_shape': output_shape}

    def __call__(self, input_value, session=None):
        """Execute forward process with the given input value."""
        session = tf.get_default_session() if session is None else session
        return session.run(
            self.output_tensor, feed_dict={self.input_tensor: input_value})

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

    def copy(self):
        """Create a new model instance with the same configuration"""
        new_model = type(self)(**self.args)
        new_model += self
        return new_model


class TFLayer(BaseLayer):
    """Add interface for scoping and summarization and output summarization"""
    def __init__(self, name, scope, args):
        super(TFLayer, self).__init__(name)

        self.args = args
        self.scope = scope

        self.input_tensor = None
        self.output_tensor = None
        self.parameter_variables = []

        self.output_summary = None
        self.parameter_summaries = None

    ###########################################################################
    def copy(self):
        return type(self)(**self.args)

    ###########################################################################
    def get_scope(self):
        if self.scope is None:
            return tf.get_variable_scope()
        return self.scope

    ###########################################################################
    def _summarize(self, tensor):
        return tf.histogram_summary(tensor.name, tensor)

    def get_parameter_summary_ops(self):
        if self.parameter_summaries is None:
            self.parameter_summaries = [
                self._summarize(t) for t in self.parameter_variables]
        return self.parameter_summaries

    def get_output_summary_op(self):
        if self.output_summary is None:
            self.output_summary = self._summarize(self.output_tensor)
        return self.output_summary
