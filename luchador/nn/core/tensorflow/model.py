from __future__ import absolute_import

import tensorflow as tf

from ..base import Model as BaseModel


class Model(BaseModel):
    def build(self, input_tensor):
        """Build the model on top of input_tensor"""
        tensor = input_tensor
        self.input_tensor = tensor
        for cfg in self.layer_configs:
            cfg.input_tensor = tensor
            with tf.variable_scope(cfg.scope or tf.get_variable_scope()):
                tensor = cfg.layer(tensor)
            cfg.output_tensor = tensor
        self.output_tensor = tensor
        return self.output_tensor
