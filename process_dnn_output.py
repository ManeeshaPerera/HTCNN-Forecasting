# 1. We need to get the batch data and the pick the approprate arrays from it
# 2. Next we need to post-process the data to inverse scale it and remove negative values
# 3. Next we need to get the average of all the forecasts - as the model is being run 5 times
# 5. Next we combine with the target data and store both in a dataframe

import numpy as np
import pandas as pd
import src.utils as util
import constants
from sklearn.preprocessing import StandardScaler
import os


def extract_non_overlapping_samples(tf_data, scaler, horizon=14):
    extracted_samples = []
    array = tf_data[0]
    for i in range(1, len(tf_data)):
        array = np.concatenate((array, tf_data[i]), axis=0)
    horizon_data = array[::horizon]
    for horizon in horizon_data:
        extracted_samples.extend(horizon.tolist())
    fc_values = scaler.inverse_transform(extracted_samples)
    fc_values[fc_values < 0] = 0
    return fc_values


def read_all_forecast_files(ts_name, num_of_iter, filepath):
    fc = []
    for num_iter in range(1, num_of_iter):
        fc_iter = pd.read_pickle(f'{filepath}/forecast_{ts_name}_iteration_{num_iter}')
        fc.append(fc_iter)

    return fc


def run_process(filepath):
    for ts in constants.TS:
        print("starting ", ts)
        data = pd.read_csv(f'ts_data/{ts}.csv', index_col=[0])
        look_back = 14 * 7  # 14 hours in to 7 days

        # train, val, test split
        train, val, test = util.split_hourly_data(data, look_back)
        dataframe_store = test[look_back:][['power']]

        scaler = StandardScaler()
        scaler.fit(train[['power']].values)

        fc_array = read_all_forecast_files(ts, 3, filepath)

        count = 0
        for iter_num_fc in fc_array:
            count = count + 1
            fc_samples = extract_non_overlapping_samples(iter_num_fc, scaler)
            dataframe_store[f'fc_{count}'] = fc_samples
        dataframe_store['average_fc'] = dataframe_store.iloc[:, 1:].mean(axis=1)

        directory_path = f'{filepath}/final_results/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        dataframe_store.to_csv(f'{directory_path}/{ts}.csv')
