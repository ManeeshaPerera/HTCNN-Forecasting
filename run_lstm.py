import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.RNN_Architectures.stacked_rnn import StackedRNN
from sklearn.preprocessing import StandardScaler
import src.utils as utils
import constants
import store_files


def run_dnn(file_index, model_name, num_of_layers, cell_dimension, epochs, lr, lookback, main_dir, model_num):
    filename = constants.TS[file_index]

    horizon = 14  # day ahead forecast
    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])
    print(filename)

    num_features = 1
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

    lstm = StackedRNN(num_of_layers, horizon, num_features, cell_dimension=cell_dimension, epochs=epochs,
                      window_generator=window_data, lr=lr)

    # run the model for 5 iterations
    for num_iter in range(1, 6):
        model = lstm.create_model()
        history = lstm.fit(model, filename)

        print("\nforecasting ==>")
        forecast, actual = lstm.forecast(model)
        store_files.save_files(model_name, filename, num_iter, history, forecast, actual, main_dir, model_num)
