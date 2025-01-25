import numpy as np
from Layers.CNN import Conv2D

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
