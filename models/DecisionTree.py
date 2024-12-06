import numpy as np
from collections import Counter


class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, info="entropy"):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.info = info

    def fit(self, x, y):
        self.tree = self._build_tree(x, y)

    def predict(self, x):
        return [self._predict_node(self.tree, sample) for sample in x]

    def _predict_node(self, tree, x):
        if tree.value is not None:
            return tree.value
        if x[tree.feature] <= tree.threshold:
            return self._predict_node(tree.left, x)
        else:
            return self._predict_node(tree.right, x)

    def _build_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        unique_classes = np.unique(y)

        if (len(unique_classes) == 1
                or n_samples < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            most_common_class = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=most_common_class)

        best_feature, best_threshold, best_gain = None, None, 0
        best_left_indices, best_right_indices = None, None

        for feature in range(n_features):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                left_indices = np.where(x[:, feature] <= threshold)[0]
                right_indices = np.where(x[:, feature] > threshold)[0]

                gain = information_gain(y, left_indices, right_indices)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_gain == 0:
            most_common_class = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=most_common_class)

        left_subtree = self._build_tree(x[best_left_indices], y[best_left_indices], depth + 1)
        right_subtree = self._build_tree(x[best_right_indices], y[best_right_indices], depth + 1)
        return DecisionNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def information_gain(self, y, left_indices, right_indices):
        func = 0
        if (self.info == "gini"):
            func =

        parent_entropy = entropy(y)
        n = len(y)
        n_left = len(left_indices)
        n_right = len(right_indices)

        if n_left == 0 or n_right == 0:
            return 0

        left_entropy = entropy(y[left_indices])
        right_entropy = entropy(y[right_indices])

        weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - weighted_entropy


