import numpy as np

from Layers.Layer import Layer

class Dropout(Layer):
    def __init__(self, dropout_rate):
        """
        Initialize the Dropout layer.
        
        Args:
            dropout_rate (float): The probability of dropping a neuron during training (0 <= dropout_rate < 1).
        """
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input, training=True):
        """
        Perform the forward pass of the Dropout layer.
        
        Args:
            input (numpy.ndarray): Input data to the layer.
            training (bool): Whether the layer is in training mode (drop neurons) or inference mode (no dropout).
        
        Returns:
            numpy.ndarray: The output after applying dropout (if training) or the same input (if inference).
        """
        if training:
            # Create a mask where neurons are dropped with probability `dropout_rate`
            self.mask = (np.random.rand(*input.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            return input * self.mask
        else:
            # During inference, no dropout is applied; return input unchanged
            return input * (1-self.dropout_rate)

    def backward(self, grad_output):
        """
        Perform the backward pass of the Dropout layer.
        
        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.
        
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Propagate gradients only through the neurons that were active during forward pass
        return grad_output * self.mask

    def update_params(self, lr):
        """
        Dropout has no learnable parameters, so this method does nothing.
        """
        pass