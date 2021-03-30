import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

import plotly.graph_objs as go
import pickle5 as pickle


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def plot_multiple_series(time_arr, series_arr, format="-", start=0, end=None):
    for trace in range(0, len(time_arr)):
        plt.plot(time_arr[trace][start:end], series_arr[trace][start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def plot_loss(history):
    fig = go.Figure(go.Scatter(y=history['loss']))
    fig.update_xaxes(title='Epochs')
    fig.update_yaxes(title='Loss (MSE)')
    fig.update_layout(height=500, width=500)
    return fig


def plot_multiple_loss(history, trace_names):
    fig = go.Figure()
    for i in range(0, len(history)):
        fig.add_trace(go.Scatter(y=history[i]['loss'], name=trace_names[i]))
    fig.update_xaxes(title='Epochs')
    fig.update_yaxes(title='Loss (MSE)')
    fig.update_layout(height=500, width=500)
    return fig


def load_pickle(file):
    with open(f'../results/{file}', 'rb') as f:
        content = pickle.load(f)
    return content


def normalise_data(data):
    n = len(data)
    train_df = data[0:int(n * 0.7)]
    val_df = data[:int(n * 0.9)]
    test_df = data

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df


def denormalise_data(train_df, data):
    train_mean = train_df.mean()
    train_std = train_df.std()

    # print(train_std, train_mean)
    new_data = (data * train_std) + train_mean
    return new_data


def process_batch(tf_batch):
    array = []
    for batch in tf_batch:
        array.extend(batch[:, 0].tolist())
    return array


def post_process_data(fc, actual, train_df):
    fc_array = np.array(process_batch(fc))
    actual_array = np.array(process_batch(actual))

    scaled_fc = denormalise_data(train_df, fc_array)
    scaled_actual = denormalise_data(train_df, actual_array)

    return scaled_fc, scaled_actual


def plot_fc(data, model_data, fc, col, look_back, show_model_data=True):
    fig = go.Figure()
    time = data.index[look_back:]
    fig.add_trace(go.Scatter(x=data.index, y=data[col].values, name='data'))
    if show_model_data:
        fig.add_trace(go.Scatter(x=time, y=model_data, name='actual'))
    fig.add_trace(go.Scatter(x=time, y=fc, name='forecast', line=dict(dash='dot', color='orange')))
    fig.update_layout(height=500, width=800, paper_bgcolor='rgba(0,0,0,0)')
    return fig


def get_samples(data, horizon):
    train_index = (int(len(data) * 0.7) // horizon) * horizon
    val_index = (int(len(data) * 0.9) // horizon) * horizon

    return data[0:train_index], data[train_index:val_index], data[val_index:]


def split_hourly_data(data, look_back):
    test = data[-((14 * constants.TEST_DAYS) + look_back):]
    val = data[-14 * (constants.VAL_DAYS + constants.TEST_DAYS): -14 * constants.TEST_DAYS]
    train = data[0:-14 * (constants.VAL_DAYS + constants.TEST_DAYS)]

    return train, val, test


def split_hourly_data_for_stat_models(data):
    test = data[-14 * constants.TEST_DAYS:]
    train = data[0:-14 * constants.TEST_DAYS]

    return train, test
