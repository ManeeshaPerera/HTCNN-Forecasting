import pickle5 as pickle
import sys
import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.RNN_Architectures.stacked_rnn import StackedRNN
from src.CNN_architectures.dilated_cnn import DilatedCNN
from sklearn.preprocessing import StandardScaler
import src.utils as utils


def save_files(model_name, filename, num_iter, history, forecast, actual, dilation=None):
    print("\nsaving files ==>")
    if (model_name == "lstm"):
        dir_name = "lstm_results"
    else:
        if (model_name == "tcn"):
            if dilation:
                dir_name = f'cnn_results/tcn/dilation_{dilation}'
            else:
                dir_name = "cnn_results/tcn"
        else:
            dir_name = "cnn_results/wavenet"

    with open(f'{dir_name}/training_loss_{filename}_iteration_{num_iter}', 'wb') as file_loss:
        pickle.dump(history.history, file_loss)

    with open(f'{dir_name}/forecast_{filename}_iteration_{num_iter}', 'wb') as file_fc:
        pickle.dump(forecast, file_fc)

    with open(f'{dir_name}/actual_{filename}_iteration_{num_iter}', 'wb') as file_ac:
        pickle.dump(actual, file_ac)


if __name__ == '__main__':
    fileindex = int(sys.argv[1])
    model_name = sys.argv[2]
    h_ts = pd.read_csv('input/ts_1h.csv', index_col=[0])
    filename = h_ts.columns[fileindex]

    OUT_STEPS = 14  # day ahead forecast
    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])
    print(filename)

    num_features = 1
    seasonality = 14
    look_back = 14 * 7  # 14 hours in to 7 days

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
    window_data = WindowGenerator(look_back, OUT_STEPS, OUT_STEPS, train_df, val_df, test_df, batch_size=128,
                                  label_columns=[col_name])

    if model_name == "lstm":
        lstm = StackedRNN(2, OUT_STEPS, num_features, cell_dimension=32, epochs=500,
                          window_generator=window_data, lr=0.001)

        # run the model for 5 iterations
        for num_iter in range(1, 6):
            model = lstm.create_model()
            history = lstm.fit(model, filename)

            print("\nforecasting ==>")
            forecast, actual = lstm.forecast(model)
            save_files(model_name, filename, num_iter, history, forecast, actual)

    if model_name == "tcn":
        num_layers = 6
        dilation_rate = 1
        # dilation_rates = [dilation_rate ** i for i in range(num_layers)]
        dilation_rates = [1, 2, 3, 4, 5, 6]
        tcn = DilatedCNN(num_layers, OUT_STEPS, num_features, n_filters=32, epochs=500, kernel_size=2,
                         dilation_rates=dilation_rates,
                         window_generator=window_data, lr=0.001)

        # run the model for 5 iterations
        for num_iter in range(1, 6):
            model = tcn.create_model()
            history = tcn.fit(model, filename)

            print("\nforecasting ==>")
            forecast, actual = tcn.forecast(model)
            save_files(model_name, filename, num_iter, history, forecast, actual, dilation_rate)
