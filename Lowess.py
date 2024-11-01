import numpy as np

import Kernel
import Knn


def smooth(x_train, y_train, kernel=Kernel.GeneralKernel(2, 3)):

    weights = np.ones(y_train.shape[0])
    for i, x_i in enumerate(x_train):
        y_i = y_train[i]

        x = np.delete(x_train, i, axis=0)
        y = np.delete(y_train, i, axis=0)

        knn = Knn.KNN(k=13)

        knn.fit(x, y)

        y_new = knn.predict(np.array([x_i]))[0]

        error = y_i != y_new
        weights[i] = kernel.process(error)

    return weights
