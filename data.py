import pandas as pd

data = pd.read_csv('input/ts_1h.csv', index_col=[0])
print(data.transpose())
