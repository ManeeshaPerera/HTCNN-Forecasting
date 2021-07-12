from src.Benchmarks._base_class import Benchmark
from sktime.forecasting.naive import NaiveForecaster


class NaiveModel(Benchmark):
    def __init__(self, train, test, horizon, seasonality, col_name):
        super().__init__(train, test, horizon, seasonality, col_name)

    def fit_naive(self):
        model = NaiveForecaster(strategy="last", sp=self.seasonality)
        model.fit(self.train_series)
        return model

    def forecast(self, model):
        forecasts = []
        fc = model.predict(self.horizon_vals)
        forecasts.extend(fc.values.tolist())
        new_train = self.train_series

        # we don't need to fit the model with the last sample data
        for sample_start_index in range(0, len(self.test_series) - self.horizon, self.horizon):
            sample_data = self.test_series[sample_start_index: sample_start_index + self.horizon]
            new_train = new_train.append(sample_data, ignore_index=True)
            model.fit(new_train)
            new_fc = model.predict(self.horizon_vals)
            forecasts.extend(new_fc.values.tolist())

        fc_df = self.test.copy()[[self.col]]
        # fc_df['average_fc'] = forecasts
        fc_df['fc'] = forecasts
        return fc_df

