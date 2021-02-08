import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_pickle(file):
    with open(f'../results/{file}', 'rb') as f:
        content = pickle.load(f)
    return content

# if __name__ == '__main__':
#     actual = load_pickle('actual_stackedRNN_region')
#     print(actual)
