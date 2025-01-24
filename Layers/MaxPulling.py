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
        self.output = np.zeros((batch_size,self.output_height, self.output_width, self.input_channels))

        for h in range(self.output_height):
            for w in range(self.output_width):
                for c in range(self.input_channels):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    h_end = h_start + self.pool_size
                    w_end = w_start + self.pool_size

                    patch = self.input[:, h_start:h_end, w_start:w_end, c]
                    self.output[:,h, w, c] = np.max(patch,axis=(1,2))

        return self.output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)

        for h in range(self.output_height):
            for w in range(self.output_width):
                for c in range(self.input_channels):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    h_end = h_start + self.pool_size
                    w_end = w_start + self.pool_size

                    patch = self.input[h_start:h_end, w_start:w_end, c]
                    max_value = np.max(patch)
                    grad_input[h_start:h_end, w_start:w_end, c] += (patch == max_value) * grad_output[h, w, c]

        return grad_input