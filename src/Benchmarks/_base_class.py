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


    # def forecast(self, model):
    #     # get the first forecast
    #     forecasts = []
    #     fc = model.predict(self.horizon_vals)
    #     forecasts.extend(fc.values.tolist())
    #     new_train = self.train_series
    #
    #     # we don't need to fit the model with the last sample data
    #     for sample_start_index in range(0, len(self.test_series) - self.horizon, self.horizon):
    #         sample_data = self.test_series[sample_start_index: sample_start_index + self.horizon]
    #         new_train.append(sample_data, ignore_index=True)
    #         model.fit(new_train)
    #         new_fc = model.predict(self.horizon_vals)
    #         forecasts.extend(new_fc.values.tolist())
    #
    #     fc_df = self.test
    #     fc_df['average_fc'] = forecasts
    #     return fc_df
