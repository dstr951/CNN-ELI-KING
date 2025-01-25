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

    def original_forward(self, input):
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
                h_start = h * self.stride
                w_start = w * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                patch = self.input[:, h_start:h_end, w_start:w_end]
                self.output[:, h, w, :] = np.sum(
                    patch[..., np.newaxis] * self.weights, axis=(1, 2, 3)
                ) + (self.biases if self.use_bias else 0)
        return self.output

    def original_backward(self, grad_output):
        # Initialize gradients
        self.grad_weights = np.zeros_like(
            self.weights)  # Shape: (kernel_size, kernel_size, input_channels, output_channels)
        self.grad_biases = np.zeros_like(self.biases) if self.use_bias else None  # Shape: (output_channels,)
        grad_input = np.zeros_like(self.input)  # Shape: (batch_size, input_height, input_width, input_channels)

        batch_size = grad_output.shape[0]  # Number of examples in the batch

        # Iterate over spatial dimensions only
        for h in range(self.output_height):
            for w in range(self.output_width):
                h_start = h * self.stride
                w_start = w * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                # Extract the patch for all examples in the batch
                patch = self.input[:, h_start:h_end, w_start:w_end,
                        :]  # Shape: (batch_size, kernel_size, kernel_size, input_channels)

                # Extract the gradient corresponding to the current output position
                grad_out_patch = grad_output[:, h, w, :]  # Shape: (batch_size, output_channels)

                # Reshape grad_out_patch for broadcasting
                grad_out_patch = grad_out_patch[:, None, None, None, :]  # Shape: (batch_size, 1, 1, 1, output_channels)

                # Update gradients for weights (sum over the batch dimension)
                self.grad_weights += np.sum(patch[..., None] * grad_out_patch,
                                            axis=0)  # Shape: (kernel_size, kernel_size, input_channels, output_channels)

                # Update gradients for biases (sum over the batch dimension)
                if self.use_bias:
                    self.grad_biases += np.sum(grad_out_patch, axis=(0, 1, 2, 3))

                # Update gradient for input
                grad_input[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.weights[None, :, :, :, :] * grad_out_patch, axis=4
                )

        # Remove padding from grad_input
        if self.padding > 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return grad_input

    def forward(self, input):
        # Optimized forward function
        self.input = np.pad(
            input, 
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 
            mode='constant'
        )
        batch_size, self.input_height, self.input_width, c_in = self.input.shape

        # Calculate output dimensions
        self.output_height = (self.input_height - self.kernel_size) // self.stride + 1
        self.output_width = (self.input_width - self.kernel_size) // self.stride + 1

        # Create an output array
        self.output = np.zeros((batch_size, self.output_height, self.output_width, self.output_channels))

        # Extract patches for all positions using striding
        patches = np.lib.stride_tricks.as_strided(
            self.input,
            shape=(
                batch_size, 
                self.output_height, 
                self.output_width, 
                self.kernel_size, 
                self.kernel_size, 
                c_in
            ),
            strides=(
                self.input.strides[0], 
                self.input.strides[1] * self.stride, 
                self.input.strides[2] * self.stride, 
                self.input.strides[1], 
                self.input.strides[2], 
                self.input.strides[3]
            ),
            writeable=False
        )  # Shape: (batch_size, output_height, output_width, kernel_size, kernel_size, input_channels)

        # Perform convolution using vectorized operations
        self.output = np.tensordot(
            patches, 
            self.weights, 
            axes=([3, 4, 5], [0, 1, 2])
        )  # Shape: (batch_size, output_height, output_width, output_channels)

        # Add biases if applicable
        if self.use_bias:
            self.output += self.biases  # Broadcasting biases over the output

        return self.output

    def backward(self, grad_output):
        # Optimized backward function
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases) if self.use_bias else None
        grad_input = np.zeros_like(self.input)

        batch_size, output_height, output_width, output_channels = grad_output.shape

        # Precompute indices for slices
        h_indices = np.arange(output_height) * self.stride
        w_indices = np.arange(output_width) * self.stride

        # Iterate over batch
        for h_idx, h_start in enumerate(h_indices):
            h_end = h_start + self.kernel_size
            for w_idx, w_start in enumerate(w_indices):
                w_end = w_start + self.kernel_size

                # Extract the patch for all examples in the batch
                patch = self.input[:, h_start:h_end, w_start:w_end, :]  # Shape: (batch_size, kernel_size, kernel_size, input_channels)

                # Extract the gradient corresponding to the current output position
                grad_out_patch = grad_output[:, h_idx, w_idx, :]  # Shape: (batch_size, output_channels)

                # Update gradients for weights (sum over the batch dimension)
                self.grad_weights += np.einsum('bijc,bo->ijco', patch, grad_out_patch)

                # Update gradients for biases (if applicable)
                if self.use_bias:
                    self.grad_biases += np.sum(grad_out_patch, axis=0)

                # Update gradient for input
                grad_input[:, h_start:h_end, w_start:w_end, :] += np.tensordot(grad_out_patch, self.weights, axes=([1], [3]))

        # Remove padding from grad_input if applicable
        if self.padding > 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return grad_input

    def update_params(self, lr):
        self.weights -= lr * self.grad_weights
        if self.use_bias:
            self.biases -= lr * self.grad_biases

# Validation Sample
if __name__ == "__main__":
    np.random.seed(42)

    # Input dimensions: (batch_size, height, width, channels)
    batch_size, height, width, input_channels = 2, 5, 5, 3
    output_channels, kernel_size, stride, padding = 4, 3, 1, 1

    # Initialize Conv2D layer
    conv = Conv2D(input_channels, output_channels, kernel_size, stride, padding, use_bias=True)

    # Create random input and gradient output
    input_data = np.random.rand(batch_size, height, width, input_channels)
    grad_output_data = np.random.rand(batch_size, (height + 2 * padding - kernel_size) // stride + 1, (width + 2 * padding - kernel_size) // stride + 1, output_channels)

    # Forward pass
    original_output = conv.original_forward(input_data)
    optimized_output = conv.forward(input_data)

    # Backward pass
    original_grad_input = conv.original_backward(grad_output_data)
    optimized_grad_input = conv.backward(grad_output_data)

    # Validation
    print("Original Forward Output Shape:", original_output.shape)
    print("Optimized Forward Output Shape:", optimized_output.shape)
    print("Forward Outputs Match:", np.allclose(original_output, optimized_output, atol=1e-6))

    print("Original Backward Grad Input Shape:", original_grad_input.shape)
    print("Optimized Backward Grad Input Shape:", optimized_grad_input.shape)
    print("Backward Grad Inputs Match:", np.allclose(original_grad_input, optimized_grad_input, atol=1e-6))
