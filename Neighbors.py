
class Neighbors:

    def __init__(self, x_train, radius, k, knn_metric):
        self.x_train = x_train
        self.radius = radius
        self.k = k
        self.knn_metric = knn_metric

    def _find_nearest_neighbors_to_point(self, x):
        points = []

        if self.radius is not None:
            for i, point in enumerate(self.x_train):
                dist = self.knn_metric.measure(x, point)
                if dist <= self.radius:
                    points.append((dist, i))

        if self.k is not None:
            for i, point in enumerate(self.x_train):
                dist = self.knn_metric.measure(x, point)
                points.append((dist, i))
            points = sorted(points, key=lambda el: el[0])[:self.k]

        distances = [point[0] for point in points]
        indices = [point[1] for point in points]

        return distances, indices

    def find_nearest_neighbors(self, x):
        distances = []
        indices = []
        for point in x:
            dist, index = self._find_nearest_neighbors_to_point(point)
            distances.append(dist)
            indices.append(index)

        return distances, indices
