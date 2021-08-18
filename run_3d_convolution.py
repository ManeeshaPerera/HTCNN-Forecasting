import sys
import pandas as pd

run = int(sys.argv[1])

import constants

SEED = constants.SEEDS[run]
import numpy as np

np.random.seed(SEED)
import tensorflow as tf

tf.random.set_seed(SEED)
import os
import pickle5 as pickle

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow import keras
from tensorflow.keras import layers
from src.CNN_Images.data_generator import DataGenerator

# Parameters
params = {'dim': (173, 192, 18),
          'batch_size': 32}

# Datasets
partition = {'train': [i for i in range(0, 5800)], 'validation': [j for j in range(5800, 5994)]}

# Generators
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

input_layer = keras.Input(shape=(173, 192, 18, 8), name=f'input_postcode')
layer_3d = tf.keras.layers.Conv3D(16, 3, activation='relu')(input_layer)
max_pool = tf.keras.layers.MaxPooling3D()(layer_3d)
layer_3d_1 = tf.keras.layers.Conv3D(16, 3, dilation_rate=(2, 2, 2), activation='relu')(max_pool)
max_pool2 = tf.keras.layers.MaxPooling3D()(layer_3d_1)
flatten_out = layers.Flatten(name='flatten_all')(max_pool2)
# prediction_layer1 = layers.Dense(100, activation='linear', name="prediction_layer1")(flatten_out)
prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)
model_3d_conv = keras.Model(inputs=input_layer, outputs=prediction_layer)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model_3d_conv.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.001),
                      metrics=[tf.metrics.MeanAbsoluteError()])
# model_3d_conv.summary()
# Train model on dataset
history = model_3d_conv.fit(training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=False,
                            workers=1, callback=[callback])

# test_data = np.load(f'swis_ts_data/img_ts/train_0.npy').reshape(1, 173, 192, 18, 8)
# print(model_3d_conv.predict(test_data))
grid_data = pd.read_csv('swis_ts_data/ts_data/grid.csv', index_col=0).iloc[18:, 0:1]
test = grid_data[-18 * constants.TEST_DAYS:]
# predict data
predictions = []
for sample in range(1, 37):
    test_data = np.load(f'swis_ts_data/img_ts/train_{sample}.npy').reshape(1, 173, 192, 18, 8)
    predictions.extend(model_3d_conv.predict(test_data)[0])

fc_df = pd.DataFrame(predictions, index=grid_data[-18 * constants.TEST_DAYS:].index, columns=['power'])
fc_df = fc_df.multiply(grid_data.rolling((18 * 6) + 1).max()).dropna()
fc_df[fc_df < 0] = 0
fc_df = fc_df.rename(columns={'power': 'fc'})
forecasts = pd.concat([test, fc_df], axis=1)
# print(df)

model_name = 'conv_3d_model'
model_new_name = f'{model_name}/{run}'
dir_path = f'swis_combined_nn_results/new_models/{model_new_name}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

forecasts.to_csv(f'{dir_path}/grid.csv')

with open(f'{dir_path}/training_loss_grid_iteration', 'wb') as file_loss:
    pickle.dump(history.history, file_loss)
