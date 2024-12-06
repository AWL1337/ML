import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from models.DecisionTree import DecisionTree
from models.RandomForest import RandomForest

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_depth(min_split, x, y):
    my_tree_depths = []
    lib_tree_depths = []
    for split in min_split:
        lib_tree = DecisionTreeClassifier(min_samples_split=split)
        my_tree = DecisionTree(min_samples=split)
        my_tree.fit(x, y)
        lib_tree.fit(x, y)
        my_tree_depths.append(my_tree.curr_max_depth)
        lib_tree_depths.append(lib_tree.get_depth())
    return my_tree_depths, lib_tree_depths


def get_f1(depths, x, y):
    my_tree_f1 = []
    lib_tree_f1 = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    for depth in depths:
        lib_tree = DecisionTreeClassifier(max_depth=depth)
        my_tree = DecisionTree(max_depth=depth)
        my_tree.fit(x_train, y_train)
        lib_tree.fit(x_train, y_train)
        my_tree_f1.append(accuracy_score(y_test, my_tree.predict(x_test)))
        lib_tree_f1.append(accuracy_score(y_test, lib_tree.predict(x_test)))
    return my_tree_f1, lib_tree_f1


def draw(data):
    x = data.drop('Rating', axis=1).to_numpy()
    y = data['Rating'].to_numpy()

    min_split = range(300, 2, -16)
    m, l = get_depth(min_split, x, y)

    plt.figure(figsize=(10, 6))
    plt.plot(min_split, m, label='my_tree')
    plt.plot(min_split, l, label='lib_tree')
    plt.xlabel('min_samples_split')
    plt.ylabel('depth')
    plt.title('Trees Depths')
    plt.legend()
    plt.grid(True)
    plt.show()

    depths = range(5, 100, 5)
    m, l = get_f1(depths, x, y)

    plt.figure(figsize=(10, 6))
    plt.plot(depths, m, label='my_tree')
    plt.plot(depths, l, label='lib_tree')
    plt.xlabel('max trees depths')
    plt.ylabel('accuracy')
    plt.title('Accuracy Score')
    plt.legend()
    plt.grid(True)
    plt.show()

