import numpy as np


class Regression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None

    def fit(self, x, y):
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        self.w = np.linalg.inv(x.T @ x + self.alpha * np.eye(x.shape[1])) @ x.T @ y

    def predict(self, x):
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        return np.sign(x @ self.w)
