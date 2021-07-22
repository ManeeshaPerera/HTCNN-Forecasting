import sys

model_func_name = sys.argv[1]
run = int(sys.argv[2])
category = int(sys.argv[3])
sample = int(sys.argv[4])

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
from constants import random_5_samples, random_25_samples, random_50_samples, random_75_samples

from src.CNN_architectures.approachA import multiple_hf_approachA
from src.CNN_architectures.approachB import grid_conv_added_at_each_TCN_multiple_hf



def create_window_data(filename, lookback=1):
    horizon = 18  # day ahead forecast

    data = pd.read_csv(filename, index_col=[0])
    print(filename)
    look_back = 18 * lookback

    train, test = utils.split_hourly_data_test_SWIS(data, look_back)

    scaler = StandardScaler()
    scaler.fit(train.values)
    train_array = scaler.transform(train.values)
    test_array = scaler.transform(test.values)

    train_df = pd.DataFrame(train_array, columns=data.columns)
    val_df = None
    test_df = pd.DataFrame(test_array, columns=data.columns)
    col_name = 'power'

    # create the dataset
    print("\ncreating final model ==>")
    window_data = WindowGenerator(look_back, horizon, horizon, train_df, val_df, test_df, batch_size=128,
                                  label_columns=[col_name])
    return window_data


def create_numpy_arrays(window_data, require_labels=False):
    input_array = []
    labels = []
    for element in window_data.as_numpy_iterator():
        for one_sample in element[0]:
            input_array.append(one_sample)

        if require_labels:
            for label in element[1]:
                labels.append(label)
    data_input = np.array(input_array, dtype=np.float32)
    data_label = np.array(labels, dtype=np.float32)
    if require_labels:
        return data_input, data_label
    else:
        return data_input


def get_samples(map_dic):
    output_values = {}
    output_labels = {}
    for key_dic, map_data in map_dic.items():
        if key_dic == 'grid':
            data_grid, label_grid = create_numpy_arrays(map_data, require_labels=True)
            output_values['input_grid'] = data_grid
            output_labels['label_grid'] = label_grid
        else:
            data_pc = create_numpy_arrays(map_data)
            output_values[f'input_postcode_{key_dic}'] = data_pc
    return output_values, output_labels['label_grid']


def run_combine_model(approach, category, pc_sample):
    sample_list = [random_5_samples, random_25_samples, random_50_samples, random_75_samples]

    window_data_pc = {}
    window_data_grid = {}

    ts_array = sample_list[category][pc_sample].copy()
    grid_str = f'grid_sample_{pc_sample}'
    ts_array.append(grid_str)

    for swis_ts in ts_array:
        if swis_ts == grid_str:
            filetoread = f'swis_ts_data/category_{category}/{swis_ts}.csv'
            window_data_grid['grid'] = create_window_data(filetoread)
        else:
            filetoread = f'swis_ts_data/ts_data/{swis_ts}.csv'
            window_data_pc[str(swis_ts)] = create_window_data(filetoread)

    window_grid = window_data_grid['grid']
    map_dic_train = window_grid.train_combine_SWIS(window_data_pc)
    map_dic_test = window_grid.test_combine_SWIS(window_data_pc)

    train_dic, label_grid = get_samples(map_dic_train)
    test_dic, label_grid_test = get_samples(map_dic_test)

    pc_list = sample_list[category][pc_sample]
    model = approach(pc_list)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    history = model.fit(train_dic, label_grid, batch_size=128, epochs=1000,
                        callbacks=[callback], shuffle=False)

    # Forecast
    lookback = 1
    data = pd.read_csv(f'swis_ts_data/category_{category}/{grid_str}.csv', index_col=[0])
    look_back = 18 * lookback

    # train, val, test split
    train, test = utils.split_hourly_data_test_SWIS(data, look_back)
    dataframe_store = test[look_back:][['power']]

    scaler = StandardScaler()
    scaler.fit(train[['power']].values)

    fc_array = []

    fc = model.predict(test_dic)

    for sample in range(0, len(fc), 18):
        fc_sample = fc[sample]
        fc_sample = scaler.inverse_transform(fc_sample)
        fc_array.extend(fc_sample)

    fc_df = pd.DataFrame(fc_array, index=data[-18 * constants.TEST_DAYS:].index, columns=['fc'])
    fc_df[fc_df < 0] = 0
    df = pd.concat([dataframe_store, fc_df], axis=1)
    return df, history


final_test_models = {'0': {'func': multiple_hf_approachA,
                           'model_name': 'method-AA',
                           'folder': 'method-AA'},
                     '1': {'func': grid_conv_added_at_each_TCN_multiple_hf,
                           'model_name': 'method-B',
                           'folder': 'method-B'}
                     }

multiple_run = final_test_models[model_func_name]['folder']
model_name = final_test_models[model_func_name]['model_name']
function_run = final_test_models[model_func_name]['func']

print("model name:", model_name)
print("seed: ", SEED)
print("run: ", run)
print("folder: ", multiple_run)

forecasts, history = run_combine_model(function_run, category, sample)

dir_path = f'swis_ts_data/category_{category}/{multiple_run}/{run}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

forecasts.to_csv(f'{dir_path}/grid_sample_{sample}.csv')

with open(f'{dir_path}/training_loss_grid_iteration_{sample}', 'wb') as file_loss:
    pickle.dump(history.history, file_loss)
