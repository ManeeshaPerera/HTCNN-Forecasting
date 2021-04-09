from src.Benchmarks._base_class import Benchmark
from prophet import Prophet
import pandas as pd


class ProphetModel(Benchmark):
    def __init__(self, train, test, horizon, seasonality, col_name, exog=False):
        super().__init__(train, test, horizon, seasonality, col_name)
        self.exog = exog

    def fit_and_forecast(self):
        fc_dfs = []
        new_train = self.train.copy()

        for sample in range(0, len(self.test), self.horizon):
            train_data = new_train.reset_index().rename(columns={'date_str': 'ds', self.col: 'y'})
            model = Prophet()
            if self.exog:
                model.add_regressor('wind')
                model.add_regressor('temperature')
                model.add_regressor('dewPoint')
                model.add_regressor('cloudCover')
                model.add_regressor('humidity')
                model.add_regressor('pressure')
                model.add_regressor('uvIndex')
            model.fit(train_data)
            test_sample = self.test[sample: sample + 14]

            if self.exog:
                test_df = test_sample.reset_index().iloc[:, [0, 2, 3, 4, 5, 6, 7, 8]].rename(columns={'date_str': 'ds'})
            else:
                test_df = test_sample.reset_index()[['date_str']].rename(columns={'date_str': 'ds'})

            fc = model.predict(test_df)
            fc_dfs.append(fc[['yhat']])
            new_train = new_train.append(test_sample)

        fc_results = pd.concat(fc_dfs)
        fc_results[fc_results < 0] = 0
        fc_results.index = self.test.index
        fc_results = fc_results.rename(columns={'yhat': 'average_fc'})

        result_df = pd.concat([self.test[['power']], fc_results], axis=1)
        return result_df
