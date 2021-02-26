import pickle5 as pickle
import sys
import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.RNN_Architectures.stacked_rnn import StackedRNN
from src.tune_parameters import TuneHyperParameters

if __name__ == '__main__':
    arguments = len(sys.argv) - 1
    horizon = int(sys.argv[1])
    col_id = int(sys.argv[2])

    OUT_STEPS = horizon
    data = pd.read_csv('input/solar_timeseries.csv', index_col=[0])
    columns = data.columns
    col_name = columns[col_id]
    pv_data = data[[col_name]]

    # train, val, test split
    n = len(pv_data)
    train_df = pv_data[0:int(n * 0.7)]
    val_df = pv_data[:int(n * 0.9)]
    test_df = pv_data

    num_features = pv_data.shape[1]
    seasonality = 288  # for 5 min data seasonality is 288

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (pv_data - train_mean) / train_std

    print("\nstarting hyper-parameter tuning")
    hp = TuneHyperParameters(OUT_STEPS, num_features, seasonality, train_df, val_df, test_df, col_name)
    model_params = hp.tune_parameters()

    # get best params
    look_back = int(model_params['look_back'])
    lr = model_params['lr']
    batch_size = int(model_params['batch_size'])
    num_layers = int(model_params['num_layers'])
    cell_dimension = int(model_params['cell_dimension'])
    epochs = int(model_params['epochs'])
    print("\nhyper-parameter tuning ended")

    # create the dataset
    print("\ncreating final model ==>")
    window_data = WindowGenerator(look_back, OUT_STEPS, OUT_STEPS, train_df, val_df, test_df, batch_size=batch_size,
                                  label_columns=[col_name])

    lstm = StackedRNN(num_layers, OUT_STEPS, num_features, cell_dimension=cell_dimension, epochs=epochs,
                      window_generator=window_data, lr=lr)
    model = lstm.create_model()
    history = lstm.fit(model)

    print("\nsaving files ==>")
    performance = lstm.evaluate_performance(model)
    print(f'performance: {performance}')
    forecast, actual = lstm.forecast(model)

    with open(f'fc_results/training_loss_{str(col_name)}', 'wb') as file_loss:
        pickle.dump(history.history, file_loss)

    with open(f'fc_results/forecast_{str(col_name)}', 'wb') as file_fc:
        pickle.dump(forecast, file_fc)

    with open(f'fc_results/actual_{str(col_name)}', 'wb') as file_ac:
        pickle.dump(actual, file_ac)
