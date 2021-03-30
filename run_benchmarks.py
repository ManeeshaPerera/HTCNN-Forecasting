import pandas as pd
import sys
import src.utils as utils
import constants as const
from src.Benchmarks.arima import ARIMA

if __name__ == '__main__':
    fileindex = int(sys.argv[1])
    method = sys.argv[2]
    filename = const.TS[fileindex]
    if fileindex > 6:
        exog = True
    else:
        exog = False

    OUT_STEPS = 14  # day ahead forecast
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
