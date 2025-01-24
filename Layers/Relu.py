class ReLU(Layer):
    def forward(self, input):
        """
        Forward pass of ReLU activation.
        Applies element-wise max(0, x).
        
        Parameters:
            input: numpy array
                Input to the layer.
                
        Returns:
            numpy array
                Output after applying ReLU.
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        """
        Backward pass of ReLU activation.
        Passes the gradient only for positive inputs.
        
        Parameters:
            grad_output: numpy array
                Gradient flowing from the next layer.
                
        Returns:
            numpy array
                Gradient flowing to the previous layer.
        """
        relu_grad = self.input > 0  # Derivative of ReLU
        return grad_output * relu_grad  # Element-wise multiplication
