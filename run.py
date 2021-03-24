import pickle5 as pickle
import sys
import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.RNN_Architectures.stacked_rnn import StackedRNN
from sklearn.preprocessing import StandardScaler
import src.utils as utils

if __name__ == '__main__':
    arguments = len(sys.argv) - 1
    fileindex = int(sys.argv[1])
    h_ts = pd.read_csv('input/ts_1h.csv', index_col=[0])
    filename = h_ts.columns[fileindex]

    OUT_STEPS = 14  # day ahead forecast
    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])
    print(data)

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

    lstm = StackedRNN(2, OUT_STEPS, num_features, cell_dimension=32, epochs=500,
                      window_generator=window_data, lr=0.001)

    # run the model for 5 iterations
    for num_iter in range(1, 6):
        model = lstm.create_model()
        history = lstm.fit(model, filename)

        print("\nforecasting ==>")
        forecast, actual = lstm.forecast(model)

        print("\nsaving files ==>")
        with open(f'lstm_results/training_loss_{filename}_iteration_{num_iter}', 'wb') as file_loss:
            pickle.dump(history.history, file_loss)

        with open(f'lstm_results/forecast_{filename}_iteration_{num_iter}', 'wb') as file_fc:
            pickle.dump(forecast, file_fc)

        with open(f'lstm_results/actual_{filename}_iteration_{num_iter}', 'wb') as file_ac:
            pickle.dump(actual, file_ac)
