from collections import Counter
import numpy as np


class Info:
    def __init__(self):
        pass

    def process(self, y):
        pass


class Entropy(Info):
    def process(self, y):
        total_count = len(y)
        if total_count == 0:
            return 0
        counts = Counter(y)
        probabilities = [count / total_count for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)


class Gini(Info):
    def process(self, y):
        total_count = len(y)
        if total_count == 0:
            return 0
        counts = Counter(y)
        probabilities = [count / total_count for count in counts.values()]
        return 1 - sum(p ** 2 for p in probabilities)
