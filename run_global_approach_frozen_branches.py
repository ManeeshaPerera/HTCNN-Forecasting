import sys

model_func_name = sys.argv[1]
run = int(sys.argv[2])
postcode = sys.argv[3]

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

from src.CNN_architectures.approachB import grid_level_branch_approachB, postcode_level_branch_approachB

from src.CNN_architectures.approachA import grid_level_branch_approachA, postcode_level_branch_approachA

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


def get_samples(grid_data, pc1, pc2, pc3, pc4, pc5, pc6):
    data_grid, label_grid = create_numpy_arrays(grid_data, require_labels=True)
    data_pc1 = create_numpy_arrays(pc1)
    data_pc2 = create_numpy_arrays(pc2)
    data_pc3 = create_numpy_arrays(pc3)
    data_pc4 = create_numpy_arrays(pc4)
    data_pc5 = create_numpy_arrays(pc5)
    data_pc6 = create_numpy_arrays(pc6)
    return data_grid, label_grid, data_pc1, data_pc2, data_pc3, data_pc4, data_pc5, data_pc6


def run_combine_model(approach, path, model_name, add_grid=True, postcode=None):
    window_grid = create_window_data(0)
    window_pc_6010 = create_window_data(7)
    window_pc_6014 = create_window_data(8)
    window_pc_6011 = create_window_data(9)
    window_pc_6280 = create_window_data(10)
    window_pc_6281 = create_window_data(11)
    window_pc_6284 = create_window_data(12)

    window_array = [window_pc_6010, window_pc_6014, window_pc_6011, window_pc_6280, window_pc_6281, window_pc_6284]

    grid, pc_1, pc_2, pc_3, pc_4, pc_5, pc_6 = window_grid.train_combine(window_array)
    # grid_val, pc_1_val, pc_2_val, pc_3_val, pc_4_val, pc_5_val, pc_6_val = window_grid.val_combine(window_array)
    grid_test, pc_1_test, pc_2_test, pc_3_test, pc_4_test, pc_5_test, pc_6_test = window_grid.test_combine(window_array)

    data_grid, label_grid, data_pc1, data_pc2, data_pc3, data_pc4, data_pc5, data_pc6 = get_samples(grid, pc_1, pc_2,
                                                                                                    pc_3, pc_4, pc_5,
                                                                                                    pc_6)

    # data_grid_val, label_grid_val, data_pc1_val, data_pc2_val, data_pc3_val, data_pc4_val, data_pc5_val, data_pc6_val = get_samples(
    #     grid_val, pc_1_val, pc_2_val,
    #     pc_3_val, pc_4_val, pc_5_val,
    #     pc_6_val)

    data_grid_test, label_grid_test, data_pc1_test, data_pc2_test, data_pc3_test, data_pc4_test, data_pc5_test, data_pc6_test = get_samples(
        grid_test, pc_1_test, pc_2_test,
        pc_3_test, pc_4_test, pc_5_test,
        pc_6_test)

    train_dic = {'input_postcode_6010': data_pc1, 'input_postcode_6014': data_pc2,
                 'input_postcode_6011': data_pc3, 'input_postcode_6280': data_pc4,
                 'input_postcode_6281': data_pc5,
                 'input_postcode_6284': data_pc6}
    # val_dic = {'input_postcode_6010': data_pc1_val, 'input_postcode_6014': data_pc2_val,
    #            'input_postcode_6011': data_pc3_val, 'input_postcode_6280': data_pc4_val,
    #            'input_postcode_6281': data_pc5_val,
    #            'input_postcode_6284': data_pc6_val}

    test_dic = {'input_postcode_6010': data_pc1_test,
                'input_postcode_6014': data_pc2_test, 'input_postcode_6011': data_pc3_test,
                'input_postcode_6280': data_pc4_test, 'input_postcode_6281': data_pc5_test,
                'input_postcode_6284': data_pc6_test}

    # CREATE MODEL
    if add_grid:
        train_dic['input_grid'] = data_grid
        # val_dic['input_grid'] = data_grid_val
        test_dic['input_grid'] = data_grid_test

    if postcode == None:
        model = approach()
    else:
        model = approach(postcode)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    # history = model.fit(train_dic, label_grid, batch_size=128, epochs=2000, validation_data=(val_dic, label_grid_val),
    #                     callbacks=[callback], shuffle=False)
    history = model.fit(train_dic, label_grid, batch_size=128, epochs=1000,
                        callbacks=[callback], shuffle=False)

    if not os.path.exists(path):
        os.makedirs(path)
    model.save(f'{path}/{model_name}')

    # Forecast
    lookback = 1
    data = pd.read_csv(f'ts_data/new/grid.csv', index_col=[0])
    look_back = 14 * lookback

    # train, val, test split
    # train, val, test = utils.split_hourly_data(data, look_back)
    train, test = utils.split_hourly_data_test(data, look_back)
    dataframe_store = test[look_back:][['power']]

    scaler = StandardScaler()
    scaler.fit(train[['power']].values)

    fc_array = []

    fc = model.predict(test_dic)

    for sample in range(0, len(fc), 14):
        fc_sample = fc[sample]
        fc_sample = scaler.inverse_transform(fc_sample)
        fc_array.extend(fc_sample)

    fc_df = pd.DataFrame(fc_array, index=data[-14 * constants.TEST_DAYS:].index, columns=['fc'])
    fc_df[fc_df < 0] = 0
    df = pd.concat([dataframe_store, fc_df], axis=1)
    return df, history


final_test_models = {'0': {'func': grid_level_branch_approachA,
                           'model_name': 'grid_level_branch_approachA'},
                     '1': {'func': grid_level_branch_approachB,
                           'model_name': 'grid_level_branch_approachB'},
                     '2': {'func': postcode_level_branch_approachA,
                           'model_name': 'postcode_level_branch_approachA'},
                     '3': {'func': postcode_level_branch_approachB,
                           'model_name': 'postcode_level_branch_approachB'}
                     }

# multiple_run = 'multiple_runs2'
multiple_run = 'pre_trained_models/CNN'
model_save_path = f'combined_nn_results/refined_models/{multiple_run}/{postcode}/saved_models'
model_name = final_test_models[model_func_name]['model_name']
function_run = final_test_models[model_func_name]['func']

print("model name:", model_name)
print("seed: ", SEED)
print("run: ", run)

if postcode == 'grid' or postcode == 'all_pc':
    pc = None
else:
    pc = postcode

model_new_name = f'{model_name}/{run}'  # this will save the models with the run info added as folder name
forecasts, history = run_combine_model(function_run, model_save_path, model_new_name, postcode=pc)

dir_path = f'combined_nn_results/refined_models/{multiple_run}/{postcode}/{model_new_name}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

forecasts.to_csv(f'{dir_path}/grid.csv')

with open(f'{dir_path}/training_loss_grid_iteration', 'wb') as file_loss:
    pickle.dump(history.history, file_loss)