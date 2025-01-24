import numpy as np

from Layers.Layer import Layer


class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        batch_size, self.input_height, self.input_width, self.input_channels = input.shape
        self.output_height = (self.input_height - self.pool_size) // self.stride + 1
        self.output_width = (self.input_width - self.pool_size) // self.stride + 1
        self.output = np.zeros((batch_size, self.output_height, self.output_width, self.input_channels))

        for h in range(self.output_height):
            for w in range(self.output_width):
                h_start = h * self.stride
                w_start = w * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size

                # Extract the patch for all channels at once
                patch = self.input[:, h_start:h_end, w_start:w_end,
                        :]  # Shape: (batch_size, pool_size, pool_size, input_channels)
                self.output[:, h, w, :] = np.max(patch, axis=(1, 2))  # Max over the spatial dimensions (h, w)

        return self.output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)

        for h in range(self.output_height):
            for w in range(self.output_width):
                h_start = h * self.stride
                w_start = w * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size

                # Extract the patch for all channels at once
                patch = self.input[:, h_start:h_end, w_start:w_end,
                        :]  # Shape: (batch_size, pool_size, pool_size, input_channels)
                max_value = np.max(patch, axis=(1, 2), keepdims=True)  # Shape: (batch_size, 1, 1, input_channels)

                # Generate the mask where the max values occur
                mask = (patch == max_value)  # Shape: (batch_size, pool_size, pool_size, input_channels)

                # Distribute the gradient output using the mask
                grad_input[:, h_start:h_end, w_start:w_end, :] += mask * grad_output[:, h:h + 1, w:w + 1,
                                                                         :]  # Broadcast grad_output

        return grad_input
