import pickle5 as pickle
import sys
import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.RNN_Architectures.stacked_rnn import StackedRNN
from src.tune_parameters import TuneHyperParameters
import src.utils as utils
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    arguments = len(sys.argv) - 1
    fileindex = int(sys.argv[1])
    h_ts = pd.read_csv('input/ts_1h.csv', index_col=[0])
    filename = h_ts.columns[fileindex]

    OUT_STEPS = 14  # day ahead forecast
    data = pd.read_csv(f'ts_data/{filename}.csv', index_col=[0])

    # train, val, test split
    train, val, test = utils.split_hourly_data(data)

    num_features = data.shape[1]
    print("number of features : ", num_features)
    seasonality = 14

    scaler = StandardScaler()
    scaler.fit(train.values)
    train_array = scaler.transform(train.values)
    val_array = scaler.transform(val.values)
    fc_array = scaler.transform(data)

    train_df = pd.DataFrame(train_array, columns=data.columns)
    val_df = pd.DataFrame(val_array, columns=data.columns)
    fc_df = pd.DataFrame(fc_array, columns=data.columns)
    col_name = 'power'

    window_data = WindowGenerator(14, OUT_STEPS, OUT_STEPS, train_df, val_df, fc_df, batch_size=128,
                                  label_columns=[col_name])
    print(window_data.train.shape)

    # print("\nstarting hyper-parameter tuning")
    # hp = TuneHyperParameters(OUT_STEPS, num_features, seasonality, train_df, val_df, fc_df, col_name)
    # model_params = hp.tune_parameters()
    #
    # # get best params
    # look_back = int(model_params['look_back'])
    # lr = model_params['lr']
    # # batch_size = int(model_params['batch_size'])
    # batch_size = 128
    # num_layers = int(model_params['num_layers'])
    # cell_dimension = int(model_params['cell_dimension'])
    # epochs = int(model_params['epochs'])
    # print("\nhyper-parameter tuning ended")
    #
    # # create the dataset
    # print("\ncreating final model ==>")
    # window_data = WindowGenerator(look_back, OUT_STEPS, OUT_STEPS, train_df, val_df, fc_df, batch_size=batch_size,
    #                               label_columns=[col_name])
    #
    # lstm = StackedRNN(num_layers, OUT_STEPS, num_features, cell_dimension=cell_dimension, epochs=epochs,
    #                   window_generator=window_data, lr=lr)
    # model = lstm.create_model()
    # history = lstm.fit(model)
    #
    # print("\nsaving files ==>")
    # performance = lstm.evaluate_performance(model)
    # print(f'performance: {performance}')
    # forecast, actual = lstm.forecast(model)
    #
    # with open(f'lstm_results/training_loss_{filename}', 'wb') as file_loss:
    #     pickle.dump(history.history, file_loss)
    #
    # with open(f'lstm_results/forecast_{filename}', 'wb') as file_fc:
    #     pickle.dump(forecast, file_fc)
    #
    # with open(f'lstm_results/actual_{filename}', 'wb') as file_ac:
    #     pickle.dump(actual, file_ac)
