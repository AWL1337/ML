import Grapics
import pandas as pd

data = pd.read_csv('games_data.csv')
Grapics.draw(data)