import numpy as np


def get_metric(name):
    if name == 'Cosine':
        return Cosine()
    if name.startswith('Minkowski_'):
        p = int(name.split('_')[1])
        return Minkowski(p)


class Metric:
    def __init__(self, name):
        self.name = name


class Cosine(Metric):
    def __init__(self):
        super().__init__("Cosine")

    def get_name(self):
        return super().name

    def measure(self, x, y):
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


class Minkowski(Metric):
    def __init__(self, p):
        self.p = p
        super().__init__("Minkowski p:{}".format(p))

    def get_name(self):
        return super().name

    def measure(self, x, y):
        return np.sum(np.abs(x - y) ** self.p) ** (1 / self.p)
