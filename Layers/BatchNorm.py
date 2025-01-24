import numpy as np

from Layers.Layer import Layer


class BatchNormalization(Layer):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, input):
        self.input = input
        self.batch_mean = np.mean(input, axis=(0, 1, 2))
        self.batch_var = np.var(input, axis=(0, 1, 2))

        self.normalized_input = (input - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        self.output = self.gamma * self.normalized_input + self.beta

        return self.output

    def backward(self, grad_output):
        N = self.input.size / self.num_features

        grad_gamma = np.sum(grad_output * self.normalized_input, axis=(0, 1, 2))
        grad_beta = np.sum(grad_output, axis=(0, 1, 2))

        grad_normalized = grad_output * self.gamma
        grad_var = np.sum(grad_normalized * (self.input - self.batch_mean) * -0.5 * (self.batch_var + self.epsilon) ** -1.5, axis=(0, 1, 2))
        grad_mean = np.sum(grad_normalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=(0, 1, 2)) + grad_var * np.sum(-2 * (self.input - self.batch_mean), axis=(0, 1, 2)) / N

        grad_input = grad_normalized / np.sqrt(self.batch_var + self.epsilon) + grad_var * 2 * (self.input - self.batch_mean) / N + grad_mean / N

        self.grad_gamma = grad_gamma
        self.grad_beta = grad_beta

        return grad_input

    def update_params(self, lr):
        self.gamma -= lr * self.grad_gamma
        self.beta -= lr * self.grad_beta