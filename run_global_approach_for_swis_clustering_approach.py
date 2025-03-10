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


from src.CNN_architectures.swis_new_architectures import concat_pc_with_grid_tcn2_for_cluster
from src.CNN_architectures.approachA import SWIS_APPROACH_A_more_layer_without_norm_cluster


def create_window_data(filename, count, lookback=1):
    horizon = 18  # day ahead forecast

    if filename == 'grid':
        data = pd.read_csv(f'swis_ts_data/ts_data/{filename}.csv', index_col=[0])
    else:
        if count == 0:
            data = pd.read_csv(f'swis_ts_data/ts_data/{filename}.csv', index_col=[0])
        else:
            data = pd.read_csv(f'swis_ts_data/ts_data/{filename}.csv', index_col=[0]).iloc[:, 0:7]
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


def run_combine_model(approach):
    window_data_pc = {}
    window_data_grid = {}

    ts_data = Clusters[cluster_num].copy()
    ts_data.append('grid')

    count=0

    for swis_ts in ts_data:
        if swis_ts == 'grid':
            window_data_grid[str(swis_ts)] = create_window_data(f'cluster_{cluster_num}', count)
        else:
            window_data_pc[str(swis_ts)] = create_window_data(swis_ts, count)
            count=+1

    window_grid = window_data_grid['grid']
    map_dic_train = window_grid.train_combine_SWIS(window_data_pc)
    map_dic_test = window_grid.test_combine_SWIS(window_data_pc)

    train_dic, label_grid = get_samples(map_dic_train)
    test_dic, label_grid_test = get_samples(map_dic_test)

    model = approach(cluster_num)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    history = model.fit(train_dic, label_grid, batch_size=128, epochs=1000,
                        callbacks=[callback], shuffle=False)

    # Forecast
    lookback = 1
    data = pd.read_csv(f'swis_ts_data/ts_data/cluster_{cluster_num}.csv', index_col=[0])
    look_back = 18 * lookback

    # train, val, test split
    train, test = utils.split_hourly_data_test_SWIS(data, look_back)
    dataframe_store = test[look_back:][['power']]

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(train[['power']].values)

    fc_array = []

    fc = model.predict(test_dic)

    for sample in range(0, len(fc), 18):
        fc_sample = fc[sample]
        # fc_sample = fc[sample].reshape(-1,1)
        fc_sample = scaler.inverse_transform(fc_sample)
        fc_array.extend(fc_sample)

    fc_df = pd.DataFrame(fc_array, index=data[-18 * constants.TEST_DAYS:].index, columns=['fc'])
    fc_df[fc_df < 0] = 0
    df = pd.concat([dataframe_store, fc_df], axis=1)
    return df, history


final_test_models = {'0': {'func': concat_pc_with_grid_tcn2_for_cluster,
                           'model_name': 'concat_pc_with_grid_tcn2_for_cluster',
                           'folder': 'new_models'},
                     '1': {'func': SWIS_APPROACH_A_more_layer_without_norm_cluster,
                           'model_name': 'SWIS_APPROACH_A_more_layer_without_norm_cluster',
                           'folder': 'new_models'}
                     }

multiple_run = final_test_models[model_func_name]['folder']
model_name = final_test_models[model_func_name]['model_name']
function_run = final_test_models[model_func_name]['func']

print("model name:", model_name)
print("seed: ", SEED)
print("run: ", run)
print("folder: ", multiple_run)

model_new_name = f'{model_name}/cluster_{cluster_num}/{run}'
forecasts, history = run_combine_model(function_run)

dir_path = f'swis_combined_nn_results/{multiple_run}/{model_new_name}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

forecasts.to_csv(f'{dir_path}/grid.csv')

with open(f'{dir_path}/training_loss_grid_iteration', 'wb') as file_loss:
    pickle.dump(history.history, file_loss)
