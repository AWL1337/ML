from collections import Counter

from DecisionTree import DecisionTree
import numpy as np


def random_subset(x, y, subset_size):
    n_samples = x.shape[0]
    indices = np.random.choice(n_samples, size=round(n_samples * subset_size), replace=False)
    return x[indices], y[indices]


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, info="entropy", min_samples=2, train_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.train_size = train_size
        self.info = info
        self.trees = []

    def fit(self, x, y):
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, info=self.info, min_samples=self.min_samples)
            x_curr, y_curr = random_subset(x, y, x)

            tree.fit(x_curr, y_curr)
            self.trees.append(tree)

    def predict(self, x):
        predictions = [tree.predict(x) for tree in self.trees]
        return [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(self.n_trees)]
