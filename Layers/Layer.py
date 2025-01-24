import numpy as np

class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def update_params(self, lr):
        pass  # For layers with parameters (e.g., Conv2D, FullyConnected)