import pandas as pd
import Graphics
import Knn
import Lowess
import ChooseParams

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('games_data.csv')

x = data.drop('Rating', axis=1).to_numpy()
y = data['Rating'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

best_params = ChooseParams.ChooseParams(x_train, y_train, x_test, y_test, x_val, y_val)

best_params = best_params.choose()

print(f"Parameters: {best_params[0]}")
print(f"Accuracy: {best_params[1]}")

Graphics.draw(x_train, y_train, x_test, y_test)

knn = Knn.KNN(k=13)
knn.fit(x_train, y_train)
y = knn.predict(x_test)
accuracy = accuracy_score(y_test, y)
print(f"Accuracy: {accuracy}")

weights = Lowess.smooth(x_train, y_train)

knn = Knn.KNN(k=13, weights=weights)
knn.fit(x_train, y_train)
y = knn.predict(x_test)
accuracy = accuracy_score(y_test, y)
print(f"Accuracy: {accuracy}")
