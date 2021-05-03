import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_network(pc):
    input_layer = keras.Input(shape=(14 * 7, 106), name=f'input_postcode_{pc}')
    x = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=1, name=f'cnn1_postcode_{pc}')(
        input_layer)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=2, name=f'cnn2_postcode_{pc}')(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=4, name=f'cnn3_postcode_{pc}')(x)
    x = layers.Flatten(name=f'flatten_postcode_{pc}')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(14, name=f'dense_postcode_{pc}')(x)
    model = keras.Model(input_layer, x)
    return model


def create_grid_network():
    input_grid = keras.Input(shape=(14 * 7, 1), name='input_grid')
    y = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=1, name='cnn1_grid')(input_grid)
    # y = layers.BatchNormalization()(y)
    # y = layers.Activation("relu")(y)
    # y = layers.Dropout(0.5)(y)
    y = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=2, name='cnn2_grid')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Flatten(name='flatten_grid')(y)
    y = layers.Dense(14, name='dense_grid')(y)
    grid = keras.Model(input_grid, y)
    return grid


def create_combine_network():
    # let's first create a network for each post code
    # 6010, 6014, 6011, 6280, 6281, 6284
    pc_6010 = create_network(6010)
    pc_6014 = create_network(6014)
    pc_6011 = create_network(6011)
    pc_6280 = create_network(6280)
    pc_6281 = create_network(6281)
    pc_6284 = create_network(6284)

    grid_network = create_grid_network()

    combinedInput = layers.concatenate(
        [grid_network.output, pc_6010.output, pc_6014.output, pc_6011.output, pc_6280.output, pc_6281.output,
         pc_6284.output])
    x = layers.Dense(14, activation="relu")(combinedInput)
    hf_model = keras.Model(
        inputs=[grid_network.input, pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
                pc_6284.input], outputs=x)

    hf_model.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam(0.0001),
                     metrics=[tf.metrics.MeanAbsoluteError()])
    return hf_model
