import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.CNN_architectures.tcn import TCN
from sklearn.preprocessing import StandardScaler
import src.utils as utils
import constants
import store_files


def run_dnn(file_index, model_name, num_of_layers, n_filters, epochs, lr, lookback, dilation_rates,
            kernel_size, main_dir, model_num):
    filename = constants.TS[file_index]

    if file_index > 6:
        num_features = 106
    else:
        num_features = 1

    horizon = 14  # day ahead forecast
    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])
    print(filename)
    # 14 hours in to 7 days
    look_back = 14 * lookback

    # train, val, test split
    train, val, test = utils.split_hourly_data(data, look_back)

    scaler = StandardScaler()
    scaler.fit(train.values)
    train_array = scaler.transform(train.values)
    val_array = scaler.transform(val.values)
    test_array = scaler.transform(test.values)

    train_df = pd.DataFrame(train_array, columns=data.columns)
    val_df = pd.DataFrame(val_array, columns=data.columns)
    test_df = pd.DataFrame(test_array, columns=data.columns)
    col_name = 'power'

    # create the dataset
    print("\ncreating final model ==>")
    window_data = WindowGenerator(look_back, horizon, horizon, train_df, val_df, test_df, batch_size=128,
                                  label_columns=[col_name])


    cnn = TCN(num_of_layers, horizon, num_features, n_filters, epochs, lr, window_data, kernel_size,
              dilation_rates, look_back)

    # run the model for 5 iterations
    for num_iter in range(1, 6):
        model = cnn.create_model()
        history = cnn.fit(model, filename)

        print("\nforecasting ==>")
        forecast, actual = cnn.forecast(model)
        store_files.save_files(model_name, filename, num_iter, history, forecast, actual, main_dir, model_num)
