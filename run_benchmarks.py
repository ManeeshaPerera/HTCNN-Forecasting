import pandas as pd
import sys
import src.utils as utils
import constants as const
from src.Benchmarks.arima import ARIMA
from src.Benchmarks.naive import NaiveModel
from src.Benchmarks.tbats import TbatsModel
from src.Benchmarks.prophet_model import ProphetModel


def run_naive(seasonality, horizon):
    for file_id in range(0, len(const.TS)):
        file_name_ts = const.TS[file_id]
        data_ts = pd.read_csv(f'ts_data/{file_name_ts}.csv', index_col=[0])
        print(file_name_ts)

        # train, val, test split
        train, test = utils.split_hourly_data_for_stat_models(data_ts)

        naive = NaiveModel(train, test, horizon, seasonality, 'power')
        model = naive.fit_naive()
        forecasts = naive.forecast(model)

        print("Saving files ==>")
        forecasts.to_csv(f'benchmark_results/naive/final_results/{filename}.csv')


if __name__ == '__main__':
    fileindex = int(sys.argv[1])
    method = sys.argv[2]
    filename = const.TS[fileindex]
    if fileindex > 6:
        exog = True
    else:
        exog = False

    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])
    print(filename)

    seasonality = const.H_SEASONALITY
    horizon = 14  # 14 hours - 1 day

    # train, val, test split
    train, test = utils.split_hourly_data_for_stat_models(data)

    if method == 'arima':
        arima = ARIMA(train, test, horizon, seasonality, 'power', exog)
        model = arima.fit_arima()
        forecasts = arima.forecast(model)

        print("Saving files ==>")
        forecasts.to_csv(f'benchmark_results/{method}/final_results/{filename}.csv')

    if method == 'naive':
        run_naive(seasonality, horizon)

    if method == 'tbats':
        tbats = TbatsModel(train, test, horizon, seasonality, 'power')
        model = tbats.fit_tbats()
        forecasts = tbats.forecast(model)

        print("Saving files ==>")
        forecasts.to_csv(f'benchmark_results/tbats/final_results/{filename}.csv')

    if method == 'prophet':
        prophet_m = ProphetModel(train, test, horizon, seasonality, 'power', exog)
        forecasts = prophet_m.fit_and_forecast()

        print("Saving files ==>")
        forecasts.to_csv(f'benchmark_results/{method}/final_results/{filename}.csv')
