import sys

time_series = int(sys.argv[1])
model_func_name = sys.argv[2]
run = int(sys.argv[3])

import constants

SEED = constants.SEEDS[run]
import numpy as np

np.random.seed(SEED)
import tensorflow as tf

tf.random.set_seed(SEED)
import os

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from sklearn.preprocessing import StandardScaler
import src.utils as utils
import pickle5 as pickle
from src.Benchmark_NNs.benchmark_nns import lstm_model_approach, conventional_CNN_approach, conventional_TCN_approach


def create_window_data(file_index, lookback=1):
    filename = constants.TS[file_index]

    horizon = 14  # day ahead forecast
    data = pd.read_csv(f'ts_data/new/{filename}.csv', index_col=[0])
    print(filename)
    # 14 hours into 1 - with the new data the days are added as features
    look_back = 14 * lookback

    # train, val, test split
    train, test = utils.split_hourly_data_test(data, look_back)
    # train, val, test = utils.split_hourly_data(data, look_back)

    scaler = StandardScaler()
    scaler.fit(train.values)
    train_array = scaler.transform(train.values)
    # val_array = scaler.transform(val.values)
    test_array = scaler.transform(test.values)

    train_df = pd.DataFrame(train_array, columns=data.columns)
    # val_df = pd.DataFrame(val_array, columns=data.columns)
    val_df = None
    test_df = pd.DataFrame(test_array, columns=data.columns)
    col_name = 'power'

    # create the dataset
    print("\ncreating final model ==>")
    window_data = WindowGenerator(look_back, horizon, horizon, train_df, val_df, test_df, batch_size=128,
                                  label_columns=[col_name])
    return window_data


def run_model(approach, window_ts, input_shape, time_series, path, model_name, file_index):
    model = approach(input_shape, time_series)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    history = model.fit(window_ts.train, batch_size=128, epochs=1000,
                        callbacks=[callback], shuffle=False)
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(f'{path}/{model_name}')

    # Forecast
    lookback = 1

    filename = constants.TS[file_index]
    data = pd.read_csv(f'ts_data/new/{filename}.csv', index_col=[0])
    look_back = 14 * lookback

    # train, val, test split
    # train, val, test = utils.split_hourly_data(data, look_back)
    train, test = utils.split_hourly_data_test(data, look_back)
    dataframe_store = test[look_back:][['power']]

    scaler = StandardScaler()
    scaler.fit(train[['power']].values)

    fc_array = []

    fc = model.predict(window_ts.test)

    for sample in range(0, len(fc), 14):
        fc_sample = fc[sample]
        fc_sample = scaler.inverse_transform(fc_sample)
        fc_array.extend(fc_sample)

    fc_df = pd.DataFrame(fc_array, index=data[-14 * constants.TEST_DAYS:].index, columns=['fc'])
    fc_df[fc_df < 0] = 0
    df = pd.concat([dataframe_store, fc_df], axis=1)
    return df, history


models = {'0': 'conventional_lstm', '1': 'conventional_cnn', '2': 'conventional_tcn'}
model_name = models[model_func_name]
model_save_path = f'{model_name}/{constants.TS[time_series]}/saved_models'

window_ts = create_window_data(time_series)
if time_series > 6:
    input_shape = (14 * 1, 14)
else:
    input_shape = (14 * 1, 7)

if model_func_name == '0':
    print('LSTM')
    print(run)
    forecasts, history = run_model(lstm_model_approach, window_ts, input_shape, time_series, model_save_path, run,
                                   time_series)

elif model_func_name == '1':
    print('CNN')
    print(run)
    forecasts, history = run_model(conventional_CNN_approach, window_ts, input_shape, time_series, model_save_path, run,
                                   time_series)

elif model_func_name == '2':
    print('TCN')
    print(run)
    forecasts, history = run_model(conventional_TCN_approach, window_ts, input_shape, time_series, model_save_path, run,
                                   time_series)

dir_path = f'{model_name}/{constants.TS[time_series]}/{run}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

forecasts.to_csv(f'{dir_path}/grid.csv')

with open(f'{dir_path}/training_loss_grid_iteration', 'wb') as file_loss:
    pickle.dump(history.history, file_loss)
