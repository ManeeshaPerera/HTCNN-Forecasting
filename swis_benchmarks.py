import pandas as pd
from src.Benchmarks.naive import NaiveModel
import src.utils as util
from constants import HORIZON_SWIS, SWIS_POSTCODES, ALL_SWIS_TS
from src.Benchmarks.arima import ARIMA
import sys


def run_naive_approach():
    for ts in ALL_SWIS_TS:
        print("running ts ==>", ts)
        pc_data = pd.read_csv(f'swis_ts_data/ts_data/{ts}.csv', index_col=0)
        train, test = util.split_train_test_statmodels_swis(pc_data)

        naive = NaiveModel(train, test, HORIZON_SWIS, HORIZON_SWIS, 'power')
        model = naive.fit_naive()
        forecasts = naive.forecast(model)
        forecasts.to_csv(f'benchmark_results/swis_benchmarks/naive/{ts}.csv')


# run_naive_approach()

# Let's check ARIMA Now - we need to adjust the data to run the arima model (our data is in the shape for DNN models so we will chnage this for ARIMA)
# for grid data we have no issue
def change_data(pc_data):
    power_data = pc_data.iloc[:, [0]]
    weather_data = pc_data.iloc[:, [7, 8, 9, 10, 11, 12, 13]]

    shift_weather = pd.DataFrame(weather_data.shift(18), columns=weather_data.columns).dropna()
    new_pc_df = pd.concat([power_data, shift_weather], axis=1).dropna()
    return new_pc_df

def run_arima(ts):
    print("running ts ==>", ts)
    exog = True
    data = pd.read_csv(f'swis_ts_data/ts_data/{ts}.csv', index_col=0)[-18*50:]
    if ts == 'grid':
        exog = False
    else:
        data = change_data(data)

    train, test = util.split_train_test_statmodels_swis(data)
    arima = ARIMA(train, test, HORIZON_SWIS, HORIZON_SWIS, 'power', exog)
    model = arima.fit_arima()
    forecasts = arima.forecast(model)
    forecasts.to_csv(f'benchmark_results/swis_benchmarks/arima/{ts}.csv')


file_index = int(sys.argv[1])
time_series = str(ALL_SWIS_TS[file_index])
run_arima(time_series)
