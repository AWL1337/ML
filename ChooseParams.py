import Knn
import optuna

from sklearn.metrics import accuracy_score


class ChooseParams:

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

    def objective(self, trial):
        metric = trial.suggest_categorical('knn_metric', ['Minkowski_1', 'Minkowski_2', 'Minkowski_3'])
        kernel = trial.suggest_categorical('knn_kernel',
                                           ['Gaussian', 'Uniform', 'General_1_1', 'General_1_2', 'General_2_1',
                                            'General_2_2'])

        use_k = trial.suggest_categorical("use_k", [True, False])

        if use_k:
            k = trial.suggest_int('k', 1, 20)
            r = None

        else:
            k = None
            r = trial.suggest_float('r', 4.0, 10.0)

        knn = Knn.KNN(k=k, radius=r, knn_metric=metric, knn_kernel=kernel)

        knn.fit(self.x_train, self.y_train)

        y = knn.predict(self.x_val)

        accuracy = accuracy_score(self.y_val, y)

        return accuracy

    def choose(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=100)
        return study.best_params, study.best_value
