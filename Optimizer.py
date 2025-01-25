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

        # Separate momentum and second-moment terms for weights and biases
        self.m_weights = None
        self.m_biases = None
        self.v_weights = None
        self.v_biases = None

    def update_params(self, weights, grad_weights, biases=None, grad_biases=None, use_bias=True):
        if self.method == 'SGD':
            # Standard SGD
            weights -= self.lr * grad_weights
            if use_bias:
                biases -= self.lr * grad_biases

        elif self.method == 'Momentum':
            # SGD with Momentum
            if self.m_weights is None:
                self.m_weights = np.zeros_like(weights)
            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
            weights -= self.lr * self.m_weights

            if use_bias:
                if self.m_biases is None:
                    self.m_biases = np.zeros_like(biases)
                self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * grad_biases
                biases -= self.lr * self.m_biases

        elif self.method == 'RMSprop':
            # RMSprop
            if self.v_weights is None:
                self.v_weights = np.zeros_like(weights)
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_weights ** 2)
            weights -= self.lr * grad_weights / (np.sqrt(self.v_weights) + self.epsilon)

            if use_bias:
                if self.v_biases is None:
                    self.v_biases = np.zeros_like(biases)
                self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (grad_biases ** 2)
                biases -= self.lr * grad_biases / (np.sqrt(self.v_biases) + self.epsilon)

        elif self.method == 'Adam':
            # Adam Optimizer
            if self.m_weights is None:
                self.m_weights = np.zeros_like(weights)
            if self.v_weights is None:
                self.v_weights = np.zeros_like(weights)

            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_weights ** 2)
            m_hat_weights = self.m_weights / (1 - self.beta1)  # Bias correction
            v_hat_weights = self.v_weights / (1 - self.beta2)  # Bias correction
            weights -= self.lr * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

            if use_bias:
                if self.m_biases is None:
                    self.m_biases = np.zeros_like(biases)
                if self.v_biases is None:
                    self.v_biases = np.zeros_like(biases)

                self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * grad_biases
                self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (grad_biases ** 2)
                m_hat_biases = self.m_biases / (1 - self.beta1)
                v_hat_biases = self.v_biases / (1 - self.beta2)
                biases -= self.lr * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

        return weights, biases
