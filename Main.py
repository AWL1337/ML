import pandas as pd
import Knn
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def objective(trial):
    metric = trial.suggest_categorical('knn_metric', ['Minkowski_1', 'Minkowski_2', 'Minkowski_3'])
    kernel = trial.suggest_categorical('knn_kernel', ['Gaussian', 'Uniform', 'General_1_1', 'General_1_2', 'General_2_1', 'General_2_2'])

    use_k = trial.suggest_categorical("use_k", [True, False])

    if use_k:
        k = trial.suggest_int('k', 1, 20)
        r = None
    else:
        k = None
        r = trial.suggest_float('r', 4.0, 10.0)

    knn = Knn.KNN(k=k, radius=r, knn_metric=metric, knn_kernel=kernel)

    knn.fit(x_train, y_train)

    y = knn.predict(x_val)

    accuracy = accuracy_score(y_val, y)

    return accuracy


data = pd.read_csv('games_data.csv')

x = data.drop('Rating', axis=1).to_numpy()
y = data['Rating'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Parameters: {study.best_params}")
print(f"Accuracy: {study.best_value}")

k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    model = Knn.KNN(k=k)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_accuracy)

    y_test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_scores.append(test_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Train Accuracy', marker='o')
plt.plot(k_values, test_scores, label='Test Accuracy', marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Dependence of Accuracy on Number of Neighbors (k)')
plt.legend()
plt.grid(True)
plt.show()
