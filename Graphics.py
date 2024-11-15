import numpy as np

from models import Regression
from models import GradientDescend
from models import SVM

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def learning_curve(model, x, y):
    train_sizes = []
    scores = []
    for i in range(1, 10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i / 10)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        scores.append(accuracy_score(y_test, y_pred))
        train_sizes.append(i / 10)

    return train_sizes, scores


def draw(data):
    x = data.drop('Rating', axis=1).to_numpy()
    y = data['Rating'].to_numpy()

    reg = learning_curve(Regression.Regression(), x, y)
    gd = learning_curve(GradientDescend.GradientDescend(), x, y)
    svm = learning_curve(SVM.SVM(), x, y)

    plt.figure(figsize=(10, 6))
    plt.plot(reg[0], reg[1], label='Regression')
    plt.plot(svm[0], svm[1], label='SVM')
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(reg[0], reg[1], color='r', label='Regression')
    plt.plot(gd[0], gd[1], color='g', label='Gradient Descend')
    plt.fill_between(gd[0], gd[1] - np.std(gd[1]), gd[1] + np.std(gd[1]), alpha=0.2, color='g')
    plt.plot(svm[0], svm[1], color='b', label='SVM')
    plt.fill_between(svm[0], svm[1] - np.std(svm[1]), svm[1] + np.std(svm[1]), alpha=0.2, color='b')
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
