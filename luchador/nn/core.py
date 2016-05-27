class Layer(object):
    def __init__(self, name):
        super(Layer, self).__init__()

        self.name = name

    def build(self, input_tensor):
        raise NotImplementedError('build method is not implemented.')


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

        self.layers = []
        self.input_tensor = None
        self.output_tensor = None

    def add(self, layer):
        self.layers.append(layer)

    def build(self, input_tensor):
        self.input_tensor = input_tensor
        for layer in self.layers:
            input_tensor = layer.build(input_tensor)
        self.output_tensor = input_tensor
        return self
