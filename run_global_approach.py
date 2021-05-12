import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from sklearn.preprocessing import StandardScaler
import src.utils as utils
import constants
import os
from src.CNN_architectures.combined_model import create_combine_network
import numpy as np
import pickle5 as pickle
import tensorflow as tf


def create_window_data(file_index, lookback):
    filename = constants.TS[file_index]

    horizon = 14  # day ahead forecast
    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])
    print(filename)
    # 14 hours in to 7 days
    look_back = 14 * lookback

    # train, val, test split
    train, test = utils.split_hourly_data_test(data, look_back)

    scaler = StandardScaler()
    scaler.fit(train.values)
    train_array = scaler.transform(train.values)
    test_array = scaler.transform(test.values)

    train_df = pd.DataFrame(train_array, columns=data.columns)
    test_df = pd.DataFrame(test_array, columns=data.columns)
    col_name = 'power'

    # create the dataset
    print("\ncreating final model ==>")
    window_data = WindowGenerator(look_back, horizon, horizon, train_df, None, test_df, batch_size=128,
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


def run_combine_model(lookback):
    window_grid = create_window_data(0, lookback)
    window_pc_6010 = create_window_data(7, lookback)
    window_pc_6014 = create_window_data(8, lookback)
    window_pc_6011 = create_window_data(9, lookback)
    window_pc_6280 = create_window_data(10, lookback)
    window_pc_6281 = create_window_data(11, lookback)
    window_pc_6284 = create_window_data(12, lookback)

    window_array = [window_pc_6010, window_pc_6014, window_pc_6011, window_pc_6280, window_pc_6281, window_pc_6284]

    grid, pc_1, pc_2, pc_3, pc_4, pc_5, pc_6 = window_grid.train_combine(window_array)
    # grid_val, pc_1_val, pc_2_val, pc_3_val, pc_4_val, pc_5_val, pc_6_val = window_grid.val_combine(window_array)
    grid_test, pc_1_test, pc_2_test, pc_3_test, pc_4_test, pc_5_test, pc_6_test = window_grid.test_combine(window_array)

    data_grid, label_grid = create_numpy_arrays(grid, require_labels=True)
    data_pc1 = create_numpy_arrays(pc_1)
    data_pc2 = create_numpy_arrays(pc_2)
    data_pc3 = create_numpy_arrays(pc_3)
    data_pc4 = create_numpy_arrays(pc_4)
    data_pc5 = create_numpy_arrays(pc_5)
    data_pc6 = create_numpy_arrays(pc_6)

    # data_grid_val, label_grid_val = create_numpy_arrays(grid_val, require_labels=True)
    # data_pc1_val = create_numpy_arrays(pc_1_val)
    # data_pc2_val = create_numpy_arrays(pc_2_val)
    # data_pc3_val = create_numpy_arrays(pc_3_val)
    # data_pc4_val = create_numpy_arrays(pc_4_val)
    # data_pc5_val = create_numpy_arrays(pc_5_val)
    # data_pc6_val = create_numpy_arrays(pc_6_val)

    data_grid_test, label_grid_test = create_numpy_arrays(grid_test, require_labels=True)
    data_pc1_test = create_numpy_arrays(pc_1_test)
    data_pc2_test = create_numpy_arrays(pc_2_test)
    data_pc3_test = create_numpy_arrays(pc_3_test)
    data_pc4_test = create_numpy_arrays(pc_4_test)
    data_pc5_test = create_numpy_arrays(pc_5_test)
    data_pc6_test = create_numpy_arrays(pc_6_test)

    model = create_combine_network()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    # history = model.fit({'input_grid': data_grid, 'input_postcode_6010': data_pc1, 'input_postcode_6014': data_pc2,
    #                      'input_postcode_6011': data_pc3, 'input_postcode_6280': data_pc4,
    #                      'input_postcode_6281': data_pc5,
    #                      'input_postcode_6284': data_pc6},
    #                     label_grid, batch_size=128, epochs=2000, validation_data=(
    #         {'input_grid': data_grid_val, 'input_postcode_6010': data_pc1_val, 'input_postcode_6014': data_pc2_val,
    #          'input_postcode_6011': data_pc3_val, 'input_postcode_6280': data_pc4_val,
    #          'input_postcode_6281': data_pc5_val,
    #          'input_postcode_6284': data_pc6_val},
    #         label_grid_val), callbacks=[callback])

    history = model.fit({'input_grid': data_grid, 'input_postcode_6010': data_pc1, 'input_postcode_6014': data_pc2,
                         'input_postcode_6011': data_pc3, 'input_postcode_6280': data_pc4,
                         'input_postcode_6281': data_pc5,
                         'input_postcode_6284': data_pc6},
                        label_grid, batch_size=128, epochs=2000, callbacks=[callback])

    # Forecast
    data = pd.read_csv(f'ts_data/grid.csv', index_col=[0])
    look_back = 14 * lookback

    # train, val, test split
    train, test = utils.split_hourly_data_test(data, look_back)
    dataframe_store = test[look_back:][['power']]

    scaler = StandardScaler()
    scaler.fit(train[['power']].values)

    fc_array = []
    fc = model.predict({'input_grid': data_grid_test, 'input_postcode_6010': data_pc1_test,
                        'input_postcode_6014': data_pc2_test, 'input_postcode_6011': data_pc3_test,
                        'input_postcode_6280': data_pc4_test, 'input_postcode_6281': data_pc5_test,
                        'input_postcode_6284': data_pc6_test})

    for sample in range(0, len(fc), 14):
        fc_sample = fc[sample]
        fc_sample = scaler.inverse_transform(fc_sample)
        fc_array.extend(fc_sample)

    fc_df = pd.DataFrame(fc_array, index=data[-14 * constants.TEST_DAYS:].index, columns=['fc'])
    fc_df[fc_df < 0] = 0
    df = pd.concat([dataframe_store, fc_df], axis=1)
    return df, history


forecasts, history = run_combine_model(7)
dir_path = 'combined_nn_results/new_models/model1'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

forecasts.to_csv(f'{dir_path}/grid.csv')

with open(f'{dir_path}/training_loss_grid_iteration', 'wb') as file_loss:
    pickle.dump(history.history, file_loss)
