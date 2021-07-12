from src.Benchmarks._base_class import Benchmark
from pmdarima import auto_arima


class ARIMA(Benchmark):
    def __init__(self, train, test, horizon, seasonality, col_name, exog=False):
        super().__init__(train, test, horizon, seasonality, col_name)
        self.exog = exog

    def fit_arima(self):
        if self.exog:
            arima = auto_arima(self.train.iloc[:, 0].values, self.train.iloc[:, 1:8].values,
                               seasonal=True,
                               m=self.seasonality, trace=True,
                               error_action='ignore',
                               suppress_warnings=True)
        else:
            arima = auto_arima(self.train.iloc[:, 0].values,
                               seasonal=True,
                               m=self.seasonality, trace=True,
                               error_action='ignore',
                               suppress_warnings=True)
        return arima

    def forecast(self, model):
        # get the first forecast
        forecasts = []

        # we don't need to fit the model with the last sample data
        for sample_start_index in range(0, len(self.test), self.horizon):
            new_data = self.test.iloc[:, 0].values[sample_start_index: sample_start_index + self.horizon]
            if self.exog is None:
                fc = model.predict(self.horizon)
                model.update(new_data)
            else:
                test_exog = self.test.iloc[:, 1:8].values[sample_start_index: sample_start_index + self.horizon]
                fc = model.predict(self.horizon, test_exog)
                model.update(new_data, exogenous=test_exog)

            forecasts.extend(fc.tolist())

        fc_df = self.test.copy()[[self.col]]
        fc_df['fc'] = forecasts
        fc_df[fc_df < 0] = 0
        return fc_df
