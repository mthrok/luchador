from __future__ import absolute_import

import tensorflow as tf

from ..base import Model as BaseModel


class Model(BaseModel):
    def build(self, input):
        """Build the model on top of input_tensor"""
        output = self.input = input()
        for cfg in self.layer_configs:
            cfg.input = output
            with tf.variable_scope(cfg.scope or tf.get_variable_scope()):
                output = cfg.layer(output)
            cfg.output = output
        self.output = output
        return self.output
