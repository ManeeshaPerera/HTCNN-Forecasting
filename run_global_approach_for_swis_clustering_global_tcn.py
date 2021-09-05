import sys

model_func_name = sys.argv[1]
run = int(sys.argv[2])
cluster_num = int(sys.argv[3])

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
from constants import Clusters, Clusters_HF

from src.Benchmark_NNs.benchmark_nns import conventional_TCN_approach

def create_window_data(filename, lookback=1):
    horizon = 18  # day ahead forecast

    data = pd.read_csv(f'swis_ts_data/ts_data/{filename}.csv', index_col=[0])
    print(filename)
    look_back = 18 * lookback

    train, test = utils.split_hourly_data_test_SWIS(data, look_back)

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


def create_numpy_arrays(window_data, require_labels=False):
    input_array = []
    labels = []
    for element in window_data.as_numpy_iterator():
        for one_sample in element[0]:
            input_array.append(one_sample)

        if require_labels:
            for label in element[1]:
                labels.append(label)
    # data_input = np.array(input_array, dtype=np.float32)
    # data_label = np.array(labels, dtype=np.float32)
    if require_labels:
        return input_array, labels
    else:
        return input_array


def get_samples(map_dic):
    output_values = []
    output_labels = []
    for key_dic, map_data in map_dic.items():
        if key_dic == 'grid':
            # we don't use this
            continue
        else:
            data_pc, label_pc = create_numpy_arrays(map_data, require_labels=True)
            print(len(data_pc))
            output_values.extend(data_pc)
            output_labels.extend(label_pc)
    # print(output_values)
    return {'input_postcode': np.array(output_values, dtype=np.float32)}, np.array(output_labels, dtype=np.float32)


def run_combine_model(approach):
    window_data_pc = {}
    window_data_grid = {}

    ts_data = Clusters[cluster_num].copy()
    ts_data.append('grid')

    for swis_ts in ts_data:
        if swis_ts == 'grid':
            window_data_grid[str(swis_ts)] = create_window_data(f'cluster_{cluster_num}')
        else:
            window_data_pc[str(swis_ts)] = create_window_data(swis_ts)

    window_grid = window_data_grid['grid']
    map_dic_train = window_grid.train_combine_SWIS(window_data_pc)
    map_dic_test = window_grid.test_combine_SWIS(window_data_pc)

    train_dic, label_grid = get_samples(map_dic_train)
    test_dic, label_grid_test = get_samples(map_dic_test)

    model = approach((18*1, 14), 'postcode')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    history = model.fit(train_dic, label_grid, batch_size=256, epochs=800,
                        callbacks=[callback], shuffle=False)

    # Forecast
    lookback = 1
    look_back = 18 * lookback
    scalers = []
    df_store = []
    fc_array = []
    for ts in ts_data:
        if ts =='grid':
            continue
        else:
            data = pd.read_csv(f'swis_ts_data/ts_data/{ts}.csv', index_col=[0])
        # train, val, test split
        train, test = utils.split_hourly_data_test_SWIS(data, look_back)
        dataframe_store = test[look_back:][['power']]
        df_store.append(dataframe_store)
        fc_array.append([])

        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        scaler.fit(train[['power']].values)
        scalers.append(scaler)

    fc = model.predict(test_dic)
    pc = 0
    sample = 0
    print(scalers)
    while sample < len(fc):
        if len(fc_array[pc]) == len(df_store[pc]):
            pc = pc+1
            sample = 631*pc
        fc_sample = fc[sample]
        # print(pc)
        # fc_sample = fc[sample].reshape(-1,1)
        pc_scaler = scalers[pc]
        # print(pc_scaler)
        fc_sample = pc_scaler.inverse_transform(fc_sample)
        fc_array[pc].extend(fc_sample)
        sample = sample + 18

    print(len(fc_array))
    print(len(fc_array[0]))
    print(len(fc_array[-1]))

    for fc_arr in range(0, len(fc_array)):
        fc_df = pd.DataFrame(fc_array[fc_arr], index=df_store[fc_arr].index, columns=['fc'])
        fc_df[fc_df < 0] = 0
        df = pd.concat([df_store[fc_arr], fc_df], axis=1)
        model_new_name = f'{model_name}/{ts_data[fc_arr]}/{run}'

        dir_path = f'swis_combined_nn_results/{multiple_run}/{model_new_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        df.to_csv(f'{dir_path}/grid.csv')


final_test_models = {'0': {'func': conventional_TCN_approach,
                           'model_name': 'conventional_TCN_approach',
                           'folder': 'new_models'}
                     }

multiple_run = final_test_models[model_func_name]['folder']
model_name = final_test_models[model_func_name]['model_name']
function_run = final_test_models[model_func_name]['func']

print("model name:", model_name)
print("seed: ", SEED)
print("run: ", run)
print("folder: ", multiple_run)


run_combine_model(function_run)
