import numpy as np

from Layers.Layer import Layer


class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, use_bias=True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        # Initialize weights and biases
        self.weights = np.random.randn(kernel_size, kernel_size, input_channels, output_channels) * 0.01
        self.biases = np.zeros(output_channels) if use_bias else None

    def forward(self, input):
        self.input = np.pad(input, ((0, 0), (self.padding, self.padding),
                                    (self.padding, self.padding), (0, 0)), mode='constant')
        batch_size, self.input_height, self.input_width, c_in = self.input.shape

        # Calculate output dimensions
        self.output_height = (self.input_height - self.kernel_size) // self.stride + 1
        self.output_width = (self.input_width - self.kernel_size) // self.stride + 1
        self.output = np.zeros((batch_size,self.output_height, self.output_width, self.output_channels))

        # Perform convolution
        for h in range(self.output_height):
            for w in range(self.output_width):
                for c in range(self.output_channels):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    h_end = h_start + self.kernel_size
                    w_end = w_start + self.kernel_size

                    patch = self.input[:, h_start:h_end, w_start:w_end]
                    self.output[:,h, w, c] = np.sum(patch * self.weights[..., c], axis=(1,2,3)) + (self.biases[c] if self.use_bias else 0)

        return self.output

    def backward(self, grad_output):
        # Initialize gradients
        self.grad_weights = np.zeros_like(self.weights)  # Shape: (kernel_size, kernel_size, input_channels, output_channels)
        self.grad_biases = np.zeros_like(self.biases) if self.use_bias else None  # Shape: (output_channels,)
        grad_input = np.zeros_like(self.input)  # Shape: (batch_size, input_height, input_width, input_channels)

        batch_size = grad_output.shape[0]  # Number of examples in the batch

        # Calculate gradients
        for h in range(self.output_height):
            for w in range(self.output_width):
                for c in range(self.output_channels):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    h_end = h_start + self.kernel_size
                    w_end = w_start + self.kernel_size

                    # Extract the patch for all examples in the batch
                    patch = self.input[:, h_start:h_end, w_start:w_end, :]  # Shape: (batch_size, kernel_size, kernel_size, input_channels)

                    # Extract the gradient corresponding to the current output position and channel
                    grad_out_patch = grad_output[:, h, w, c]  # Shape: (batch_size,)

                    # Reshape grad_out_patch for broadcasting
                    grad_out_patch = grad_out_patch[:, None, None, None]  # Shape: (batch_size, 1, 1, 1)

                    # Update gradients for weights (sum over the batch dimension)
                    self.grad_weights[..., c] += np.sum(patch * grad_out_patch, axis=0)

                    # Update gradients for biases (sum over the batch dimension)
                    if self.use_bias:
                        self.grad_biases[c] += np.sum(grad_out_patch)

                    # Update gradient for input
                    grad_input[:, h_start:h_end, w_start:w_end, :] += (
                        self.weights[..., c][None, :, :, :] * grad_out_patch
                    )

        # Remove padding from grad_input
        if self.padding > 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return grad_input


    def update_params(self, lr):
        self.weights -= lr * self.grad_weights
        if self.use_bias:
            self.biases -= lr * self.grad_biases