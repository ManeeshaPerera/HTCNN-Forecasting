import tensorflow as tf
import numpy as np
import pandas as pd
import pickle5 as pickle
# import pickle
strategy = tf.distribute.MirroredStrategy()


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


data = pd.read_pickle('input/hf_data')
series = data['grid'][0:50000].values
time = data.index[0:50000].values

split_time = int(50000 * 0.7)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 12
batch_size = 100
shuffle_buffer_size = 1000

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                               input_shape=[None]),
        tf.keras.layers.LSTM(40, return_sequences=True),
        tf.keras.layers.LSTM(40),
        tf.keras.layers.Dense(1),
        # tf.keras.layers.Lambda(lambda x: x * 1000.0)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=["mae"])
    model.summary()
history = model.fit(dataset, epochs=500)

with open('results/trainHistoryDict1', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]
with open('results/forecast_region1', 'wb') as file_forecast:
    pickle.dump(results, file_forecast)
