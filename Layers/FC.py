import numpy as np

from Layers.Layer import Layer
from Optimizer import Optimizer
import Consts


class FullyConnected(Layer):
    def __init__(self, input_tuple, output_size, use_bias=True):
        self.input_size = input_tuple
        self.output_size = output_size
        self.use_bias = use_bias

        # Initialize weights and biases
        self.weights = np.random.randn(input_tuple[1] , output_size[1]) * 0.01
        self.biases = np.zeros(output_size[1]) if use_bias else None

        self.optimizer = Optimizer(Consts.OPTIMIZER_METHOD, Consts.LEARNING_RATE)

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights)
        if self.use_bias:
            self.output += self.biases
        return self.output

    def backward(self, grad_output):
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0) if self.use_bias else None
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def update_params(self, lr):
        self.optimizer.update_params(self.weights, self.grad_weights, self.biases, self.grad_biases, use_bias=True)