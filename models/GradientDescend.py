import numpy as np


class GradientDescend:
    def __init__(self, learning_rate=0.01, max_iter=1000, alpha=0.5, lambda1=0.1, lambda2=0.1, loss='sigmoid'):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss = loss
        self.w = None
        self.bias = 0

    def margin(self, x, y):
        return y * (x @ self.w + self.bias)

    def gradient(self, x, y):
        dL = 0
        margin = self.margin(x, y)
        if self.loss == 'lda':
            dL = -2 * (1 - margin)
        elif self.loss == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-margin))
            dL = -2 * sigmoid * (1 - sigmoid)
        elif self.loss == 'exponential':
            dL = -np.exp(-margin)

        dW = x.T @ (dL * y) / x.shape[0]
        dB = np.mean(dL * y)

        dW += self.alpha * (self.lambda1 * np.sign(self.w) + self.lambda2 * self.w)

        return dW, dB

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            dW, dB = self.gradient(x, y)
            self.w -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

    def predict(self, x):
        return np.sign(x @ self.w + self.bias)
