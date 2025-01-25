import numpy as np


class Optimizer:
    METHOD_SGD = 'SGD'
    METHOD_MOMENTUM = 'Momentum'
    METHOD_RMS_PROP = 'RMSprop'
    METHOD_ADAM = 'Adam'

    def __init__(self, method='SGD', lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = method
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # Momentum for Adam/RMSprop
        self.v = None  # RMSprop or Adam second moment

    def update_params(self, weights, grad_weights, biases=None, grad_biases=None, use_bias=True):
        if self.method == 'SGD':
            # Standard SGD
            weights -= self.lr * grad_weights
            if use_bias:
                biases -= self.lr * grad_biases

        elif self.method == 'Momentum':
            # SGD with Momentum
            if self.m is None:
                self.m = np.zeros_like(weights)
            self.m = self.beta1 * self.m + self.lr * grad_weights
            weights -= self.m
            if use_bias:
                if self.m is None:
                    self.m = np.zeros_like(biases)
                self.m = self.beta1 * self.m + self.lr * grad_biases
                biases -= self.m

        elif self.method == 'RMSprop':
            # RMSprop
            if self.v is None:
                self.v = np.zeros_like(weights)
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_weights ** 2)
            weights -= self.lr * grad_weights / (np.sqrt(self.v) + self.epsilon)
            if use_bias:
                if self.v is None:
                    self.v = np.zeros_like(biases)
                self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_biases ** 2)
                biases -= self.lr * grad_biases / (np.sqrt(self.v) + self.epsilon)

        elif self.method == 'Adam':
            # Adam Optimizer
            if self.m is None:
                self.m = np.zeros_like(weights)
            if self.v is None:
                self.v = np.zeros_like(weights)

            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_weights
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_weights ** 2)
            m_hat = self.m / (1 - self.beta1)  # Bias correction
            v_hat = self.v / (1 - self.beta2)  # Bias correction
            weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            if use_bias:
                if self.m is None:
                    self.m = np.zeros_like(biases)
                if self.v is None:
                    self.v = np.zeros_like(biases)
                self.m = self.beta1 * self.m + (1 - self.beta1) * grad_biases
                self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_biases ** 2)
                m_hat = self.m / (1 - self.beta1)
                v_hat = self.v / (1 - self.beta2)
                biases -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights, biases