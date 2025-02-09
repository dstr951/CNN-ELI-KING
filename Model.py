import numpy as np

from Consts import *
from Layers.Dropout import Dropout


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        """
        Performs a forward pass through all layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, loss_grad):
        """
        Performs a backward pass through all layers in reverse order.
        """
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def update_params(self, lr):
        """
        Updates parameters of all layers that have parameters.
        """
        for layer in self.layers:
            layer.update_params(lr)

    def inference(self, input):
        for layer in self.layers:
            if isinstance(layer,Dropout):
                input = layer.forward(input, training=False)
            else:
                input = layer.forward(input)
        return np.argmax(input, axis=1) + 1