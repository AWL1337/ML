from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import Knn


def draw(x_train, y_train, x_test, y_test):
    k_values = range(1, 21)
    train_scores = []
    test_scores = []

    for k in k_values:
        knn = Knn.KNN(k=k)
        knn.fit(x_train, y_train)

        y_train_pred = knn.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_scores.append(train_accuracy)

        y_test_pred = knn.predict(x_test)
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
