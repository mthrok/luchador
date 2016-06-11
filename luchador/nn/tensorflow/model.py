from __future__ import absolute_import

import tensorflow as tf

from ..core import BaseModel


class Model(BaseModel):
    """Add interface for concatenation and summarization to BaseLayer"""
    def build(self, input_tensor):
        """Build the model on top of input_tensor"""
        self.input_tensor = input_tensor
        tensor = input_tensor
        for layer_config in self.layer_configs:
            layer = layer_config['layer']
            scope = layer_config['scope'] or tf.get_variable_scope()

            layer_config['input_tensor'] = tensor
            with tf.variable_scope(scope):
                tensor = layer(tensor)
            layer_config['output_tensor'] = tensor
        self.output_tensor = tensor
        return self.output_tensor

    ###########################################################################
    def get_parameter_variables(self):
        """Get tf.Variable objects consisting the parameters of this model"""
        ret = []
        for layer_config in self.layer_configs:
            layer = layer_config['layer']
            for variable in layer.parameter_variables.values():
                ret.append(variable)
        return ret

    def get_output_tensors(self):
        """Get tf.Tensor objects which represent the output of each layer"""
        ret = []
        for layer_config in self.layer_configs:
            tensor = layer_config['output_tensor']
            ret.append(tensor)
        return ret
