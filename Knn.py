from collections import defaultdict

import Metric
import Kernel
import Neighbors


class KNN:
    def __init__(self, radius=None, k=None, weights=None,
                 knn_metric=Metric.Minkowski(2), knn_kernel=Kernel.GaussianKernel()):
        self.radius = radius
        self.k = k
        self.weights = weights
        self.knn_metric = knn_metric
        self.knn_kernel = knn_kernel
        self.x_train = []
        self.y_train = []
        self.neighbors = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.neighbors = Neighbors.Neighbors(self.x_train, self.radius, self.k, self.knn_metric)

    def predict(self, x):
        dist, index = self.neighbors.find_nearest_neighbors(x)
        predictions = []

        for i, point in enumerate(x):
            w = self.knn_kernel.process(dist[i])

            if self.weights is not None:
                a_w = [self.weights[idx] for idx in index[i]]
                w = [a_w * w for a_w, w in zip(a_w, w)]

            answers = [self.y_train[i] for i in index[i]]

            s = defaultdict(int)
            for ans, w in zip(answers, w):
                s[ans] += w

            prediction = max(s, key=s.get)

            predictions.append(prediction)

        return predictions
