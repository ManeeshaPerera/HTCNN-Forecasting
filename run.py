import pickle5 as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.RNN_Architectures.stacked_rnn import StackedRNN

if __name__ == '__main__':
    # arguments = len(sys.argv) - 1
    # horizon = int(sys.argv[1])
    # col_id = int(sys.argv[2])

    OUT_STEPS = 3
    data = pd.read_pickle('input/ts_1h')
    data = data.resample('1D').mean()
    # columns = data.columns
    # col_name = columns[col_id]
    pv_data = data[['grid']]
    # print("\ncolumn name: ", col_name)

    # train, val, test split
    # n = len(pv_data)
    # train_df = pv_data[0:int(n * 0.7)]
    # val_df = pv_data[:int(n * 0.9)]
    # test_df = pv_data
    #
    num_features = pv_data.shape[1]
    val_start = pd.Timestamp("2021-01-31")
    df = pv_data.reset_index()
    train = df.loc[df['date_str'] < val_start][['grid']]
    test_df = df[['grid']]
    # print(train)
    train_df = train[0:-50]
    val_df = train[-50:]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (pv_data - train_mean) / train_std
    print(train_df, val_df, test_df)

    # create the dataset
    print("\ncreating final model ==>")
    window_data = WindowGenerator(12, OUT_STEPS, OUT_STEPS, train_df, val_df, test_df, batch_size=32,
                                  label_columns=['grid'])
    # print(window_data.test_df)

    lstm = StackedRNN(2, OUT_STEPS, num_features, cell_dimension=32, epochs=300,
                      window_generator=window_data, lr=0.001)
    model = lstm.create_model()
    history = lstm.fit(model)
    #
    # print("\nsaving files ==>")
    forecast, actual = lstm.forecast(model)
    # #
    # # with open(f'fc_results/training_loss_{col_name}', 'wb') as file_loss:
    # #     pickle.dump(history.history, file_loss)
    #
    with open(f'fc_results/forecast_rnn', 'wb') as file_fc:
        pickle.dump(forecast, file_fc)

    # with open(f'fc_results/actual_{col_name}', 'wb') as file_ac:
    #     pickle.dump(actual, file_ac)
