import numpy as np

from Layers.Layer import Layer


class Softmax(Layer):
    def forward(self, input):
        """
        Forward pass of Softmax activation.
        Converts logits into probabilities.
        
        Parameters:
            input: numpy array
                Input to the layer (logits).
                
        Returns:
            numpy array
                Softmax probabilities.
        """
        # Subtract max for numerical stability
        exps = np.exp(input - np.max(input, axis=-1, keepdims=True))
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        """
        Backward pass of Softmax activation.
        Computes the gradient of the loss w.r.t. the input.
        
        Parameters:
            grad_output: numpy array
                Gradient flowing from the next layer.
                
        Returns:
            numpy array
                Gradient flowing to the previous layer.
        """
        batch_size, num_classes = self.output.shape
        grad_input = np.zeros_like(self.output)

        for i in range(batch_size):
            # Jacobian matrix of softmax for the current sample
            softmax_output = self.output[i].reshape(-1, 1)  # Shape: (num_classes, 1)
            jacobian = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)  # Shape: (num_classes, num_classes)

            # Compute gradient for the current sample
            grad_input[i] = np.dot(jacobian, grad_output[i])

        return grad_input
