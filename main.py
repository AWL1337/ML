import pandas as pd
import ChooseParams
from sklearn.model_selection import train_test_split


data = pd.read_csv('games_data.csv')

x = data.drop('Rating', axis=1).to_numpy()
y = data['Rating'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

params = ChooseParams.ChooseParams(x_train, y_train, x_test, y_test, x_val, y_val)

best_params = params.choose_regression()

print("Regression")
print(f"Parameters: {best_params[0]}")
print(f"Accuracy: {best_params[1]}")

best_params = params.choose_gd()

print("GradientDescend")
print(f"Parameters: {best_params[0]}")
print(f"Accuracy: {best_params[1]}")

best_params = params.choose_swm()

print("SVM")
print(f"Parameters: {best_params[0]}")
print(f"Accuracy: {best_params[1]}")
