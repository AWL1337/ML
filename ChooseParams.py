from models import Regression
from models import GradientDescend
from models import SVM
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

    def objective_regression(self, trial):
        alpha = trial.suggest_float('alpha', 0.0, 2.0)

        r = Regression.Regression(alpha=alpha)

        r.fit(self.x_train, self.y_train)

        y = r.predict(self.x_val)

        accuracy = accuracy_score(self.y_val, y)

        return accuracy

    def choose_regression(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective_regression, n_trials=100)
        return study.best_params, study.best_value

    def objective_gd(self, trial):

        loss = trial.suggest_categorical('loss', ['log', 'lda', 'exp', 'sigmoid'])

        alpha = trial.suggest_float('alpha', 0.0, 2.0)

        learning_rate = trial.suggest_float('learning_rate', 0.0, 0.99)

        max_iter = trial.suggest_int('max_iter', 100, 1000)

        lambda1 = trial.suggest_float('lambda1', 0.0, 2.0)
        lambda2 = trial.suggest_float('lambda2', 0.0, 2.0)

        r = GradientDescend.GradientDescend(loss=loss, alpha=alpha, learning_rate=learning_rate,
                                            max_iter=max_iter, lambda1=lambda1, lambda2=lambda2)

        r.fit(self.x_train, self.y_train)

        y = r.predict(self.x_val)

        accuracy = accuracy_score(self.y_val, y)

        return accuracy

    def choose_gd(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective_gd, n_trials=200)
        return study.best_params, study.best_value
