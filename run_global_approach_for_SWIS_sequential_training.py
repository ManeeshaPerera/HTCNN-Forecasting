import sys

model_func_name = sys.argv[1]
run = int(sys.argv[2])

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
from constants import ALL_SWIS_TS, SWIS_POSTCODES

from src.CNN_architectures.approachB_sequentional_training import sequentional_training_approach


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


def run_combine_model():
    window_data_pc = {}
    window_data_grid = {}

    for swis_ts in ALL_SWIS_TS:
        if swis_ts == 'grid':
            window_data_grid[str(swis_ts)] = create_window_data(swis_ts)
        else:
            window_data_pc[str(swis_ts)] = create_window_data(swis_ts)

    window_grid = window_data_grid['grid']
    map_dic_train = window_grid.train_combine_SWIS(window_data_pc)
    map_dic_test = window_grid.test_combine_SWIS(window_data_pc)

    train_dic_start, label_grid_start = get_samples(map_dic_train)
    test_dic_start, _ = get_samples(map_dic_test)

    grid_valus = train_dic_start['input_grid']
    new_grid_input = []
    new_pc_input = []
    new_label = []

    for grid_index in range(0, len(grid_valus)):
        for pc in ALL_SWIS_TS[1:]:
            new_grid_input.append(grid_valus[grid_index])
            new_pc_input.append(train_dic_start[f'input_postcode_{pc}'][grid_index])
            new_label.append(label_grid_start[grid_index])

    grid_valus_test = test_dic_start['input_grid']
    new_test_grid = []
    new_test_pc = []

    for grid_index_test in range(0, len(grid_valus_test)):
        for pc in ALL_SWIS_TS[1:]:
            new_test_grid.append(grid_valus_test[grid_index_test])
            new_test_pc.append(test_dic_start[f'input_postcode_{pc}'][grid_index_test])

    train_dic = {'input_grid': np.array(new_grid_input, dtype=np.float32),
                 'input_pc': np.array(new_pc_input, dtype=np.float32)}
    label_grid = np.array(new_label, dtype=np.float32)
    test_dic = {'input_grid': np.array(new_test_grid, dtype=np.float32),
                'input_pc': np.array(new_test_pc, dtype=np.float32)}

    model = sequentional_training_approach()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    history = model.fit(train_dic, label_grid, batch_size=128*5, epochs=1000,
                        callbacks=[callback], shuffle=False)

    # Forecast
    lookback = 1
    data = pd.read_csv(f'swis_ts_data/ts_data/grid.csv', index_col=[0])
    look_back = 18 * lookback

    # train, val, test split
    train, test = utils.split_hourly_data_test_SWIS(data, look_back)
    dataframe_store = test[look_back:][['power']]

    scaler = StandardScaler()
    scaler.fit(train[['power']].values)

    fc_array = []

    fc = model.predict(test_dic)
    pc_length = len(ALL_SWIS_TS[1:])

    for sample in range(pc_length - 1, len(fc), pc_length * 18):
        fc_sample = fc[sample]
        fc_sample = scaler.inverse_transform(fc_sample)
        fc_array.extend(fc_sample)

    fc_df = pd.DataFrame(fc_array, index=data[-18 * constants.TEST_DAYS:].index, columns=['fc'])
    fc_df[fc_df < 0] = 0
    df = pd.concat([dataframe_store, fc_df], axis=1)
    print(df)
    return df, history


final_test_models = {'0': {'func': sequentional_training_approach,
                           'model_name': 'sequentional_training_approach',
                           'folder': 'approachB'}
                     }

multiple_run = final_test_models[model_func_name]['folder']
model_save_path = f'swis_combined_nn_results/{multiple_run}/saved_models'
model_name = final_test_models[model_func_name]['model_name']
function_run = final_test_models[model_func_name]['func']

print("model name:", model_name)
print("seed: ", SEED)
print("run: ", run)
print("folder: ", multiple_run)

model_new_name = f'{model_name}/{run}'  # this will save the models with the run info added as folder name
forecasts, history = run_combine_model()

dir_path = f'swis_combined_nn_results/{multiple_run}/{model_new_name}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

forecasts.to_csv(f'{dir_path}/grid.csv')

with open(f'{dir_path}/training_loss_grid_iteration', 'wb') as file_loss:
    pickle.dump(history.history, file_loss)
