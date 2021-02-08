import pickle5 as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
from src.WindowGenerator.window_generator import WindowGenerator
from src.RNN_Architectures.stacked_rnn import StackedRNN


def plot_distribution(df_std):
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(pv_data.keys(), rotation=90)
    plt.show()


if __name__ == '__main__':
    arguments = len(sys.argv) - 1
    model_name = sys.argv[1]
    horizon = int(sys.argv[2])

    OUT_STEPS = horizon
    # read the data (using pickle5 due the compatibility issues with Spartan)
    # with open('input/hf_data', 'rb') as f:
    #     pv_data = pickle.load(f)[['grid']]
    pv_data = pd.read_csv('input/rnn_data.csv', index_col=[0])[['grid']]

    # train, val, test split
    n = len(pv_data)
    train_df = pv_data[0:int(n * 0.7)]
    val_df = pv_data[:int(n * 0.9)]
    test_df = pv_data

    num_features = pv_data.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (pv_data - train_mean) / train_std
    # plot the distribution of features
    # plot_distribution(df_std)

    # create the dataset
    window_data = WindowGenerator(288, OUT_STEPS, OUT_STEPS, train_df, val_df, test_df, batch_size=2016,
                                  label_columns=['grid'])
    for example_inputs, example_labels in window_data.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    lstm = StackedRNN(1, OUT_STEPS, num_features, epochs=500, window_generator=window_data)
    lstm.create_model()
    history = lstm.compile_and_fit()

    performance = lstm.evaluate()
    print(f'performance: {performance}')
    forecast, actual = lstm.forecast()

    with open(f'results/training_loss_{model_name}', 'wb') as file_loss:
        pickle.dump(history.history, file_loss)

    with open(f'results/forecast_{model_name}', 'wb') as file_fc:
        pickle.dump(forecast, file_fc)

    with open(f'results/actual_{model_name}', 'wb') as file_ac:
        pickle.dump(actual, file_ac)
