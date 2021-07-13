from constants import ALL_SWIS_TS
from os import listdir
import pandas as pd


arima_success = []
for f in listdir('benchmark_results/swis_benchmarks/arima'):
    pc = f.split('.')[0]
    arima_success.append(pc)

for ts in range(0, len(ALL_SWIS_TS)):
    time_Series = ALL_SWIS_TS[ts]
    if str(time_Series) not in arima_success:
        print(ts, time_Series)


# data = pd.read_csv('swis_ts_data/ts_data/6102.csv', index_col=0)
# print(data)