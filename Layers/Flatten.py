from Layers.Layer import Layer


class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)  # Flatten everything except the batch dimension

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)