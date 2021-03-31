import numpy as np
import pandas as pd


class Benchmark:
    def __init__(self, train, test, horizon, seasonality, col_name):
        self.train = train
        self.test = test
        self.horizon = horizon
        self.seasonality = seasonality
        self.horizon_vals = np.arange(horizon) + 1
        self.train_series = pd.Series(train[col_name].values)
        self.test_series = pd.Series(test[col_name].values)
        self.col = col_name