import pandas as pd
from sklearn.model_selection import train_test_split

import Knn

data = pd.read_csv('games_data.csv')

x = data.drop('Rating', axis=1).to_numpy()
y = data['Rating'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn = Knn.KNN(k=5)

knn.fit(x_train, y_train)


print(knn.predict(x_test))
