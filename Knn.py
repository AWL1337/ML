from collections import defaultdict

import Metric
import Kernel
import Neighbors


class KNN:
    def __init__(self, radius=None, k=None, weights=None,
                 knn_metric='Minkowski_2', knn_kernel='Gaussian'):
        self.radius = radius
        self.k = k
        self.weights = weights
        self.knn_metric = Metric.get_metric(knn_metric)
        self.knn_kernel = Kernel.get_kernel(knn_kernel)
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
            weights = self.knn_kernel.process(dist[i])

            if self.weights is not None:
                a_w = [self.weights[idx] for idx in index[i]]
                weights = [a_w * w for a_w, w in zip(a_w, weights)]

            answers = [self.y_train[k] for k in index[i]]

            s = defaultdict(int)
            for ans, w in zip(answers, weights):
                s[ans] += w

            prediction = max(s, key=s.get)

            predictions.append(prediction)

        return predictions
