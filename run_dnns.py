import pickle5 as pickle
import sys
import pandas as pd
import constants
from src.DNN.lstm import LSTMModel
import os

if __name__ == '__main__':
    fileindex = int(sys.argv[1])
    model_name = sys.argv[2]
    filename = constants.TS[fileindex]

    if fileindex > 6:
        exog = True
        epochs = 500
        lr = 0.0001
        layers = 3
        cell_dim = 64
    else:
        exog = False
        epochs = 200
        lr = 0.001
        layers = 1
        cell_dim = 16

    horizon = 14  # day ahead forecast
    num_lags = 14 * 3
    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])
    print(filename)

    lstm = LSTMModel(horizon, num_lags, data, epochs, lr, layers, cell_dim, exog)
    train_X, train_Y, val_X, val_Y, test_X, test_Y = lstm.get_train_val_test()

    for i in range(1, 2):
        history, model = lstm.compile_and_fit_lstm(train_X, train_Y, val_X, val_Y)
        forecasts = lstm.get_forecast(test_X, model)
        print("Saving files ==>")
        directory_path = f'dnn_results/{model_name}/{filename}'

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        forecasts.to_csv(f'{directory_path}/forecasts_iteration_{i}.csv')
        with open(f'{directory_path}/training_loss_iteration_{i}', 'wb') as file_loss:
            pickle.dump(history.history, file_loss)

