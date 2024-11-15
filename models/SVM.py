import numpy as np


class SVM:
    def __init__(self, C=1.047, learning_rate=0.228, max_iters=954, kernel='rfb_0.3'):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.kernel = kernel
        self.alpha = None
        self.b = 0
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

        self.alpha = np.zeros(x.shape[0])

        K = self.kernel_matrix(x)

        for _ in range(self.max_iters):
            for i in range(x.shape[0]):
                self.alpha[i] = self.alpha[i] - self.learning_rate * self.gradient(K, y, i)
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)

        self.b = np.mean(y - np.dot(K, (self.alpha * y)))

    def kernel_matrix(self, x):
        K = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                K[i, j] = self.kernelf(x[i], x[j])
        return K

    def kernelf(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel.startswith('polynomial_'):
            degree = int(self.kernel.split('_')[1])
            return (np.dot(x1, x2) + 1) ** degree
        elif self.kernel.startswith('rfb_'):
            gamma = float(self.kernel.split('_')[1])
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel.startswith('sigmoid_'):
            gamma, r = map(float, self.kernel.split('_')[1:])
            return np.tanh(gamma * np.dot(x1, x2) + r)

    def gradient(self, K, y, i):
        return np.sum(self.alpha * y * K[:, i]) - 1

    def predict(self, x):

        K_test = np.zeros((x.shape[0], self.x_train.shape[0]))
        for i in range(x.shape[0]):
            for j in range(self.x_train.shape[0]):
                K_test[i, j] = self.kernelf(x[i], self.x_train[j])

        y_predict = np.dot(K_test, (self.alpha * self.y_train)) + self.b
        return np.sign(y_predict)