import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn


def create_network(pc):
    input_layer = keras.Input(shape=(14 * 1, 14), name=f'input_postcode_{pc}')
    # x = layers.Conv1D(kernel_size=4, padding='same', filters=32, name=f'cnn1_postcode_{pc}')(
    #     input_layer)
    # x = layers.MaxPooling1D(padding='same', strides=1)(x)
    # x = layers.Conv1D(kernel_size=4, padding='same', filters=32, name=f'cnn2_postcode_{pc}')(
    #     x)
    # x = layers.MaxPooling1D(padding='same', strides=1)(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=2, name=f'cnn2_postcode_ly1_{pc}')(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=4, name=f'cnn3_postcode_ly2_{pc}')(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=8, name=f'cnn3_postcode_ly3_{pc}')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Flatten(name=f'flatten_postcode_{pc}')(x)
    # x = layers.Dense(14, name=f'dense_postcode_{pc}')(x)
    # x = layers.Dense(14, name=f'dense_postcode_{pc}')(input_layer)
    # model = keras.Model(input_layer, x)
    return input_layer
    # return input_layer


def create_grid_network():
    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')
    # y = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=1, name='cnn1_grid')(input_grid)
    # y = layers.BatchNormalization()(y)
    # y = layers.Activation("relu")(y)
    # y = layers.Dropout(0.5)(y)
    # y = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=2, name='cnn2_ly1_grid')(y)
    # y = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=4, name='cnn2_ly2_grid')(y)
    # y = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=8, name='cnn2_ly3_grid')(y)
    # y = layers.BatchNormalization()(y)
    # y = layers.Activation("relu")(y)
    # y = layers.Dropout(0.5)(y)
    # y = layers.Flatten(name='flatten_grid')(y)
    # y = layers.Dense(14, name='dense_grid')(y)
    # y = layers.Dense(14, name='dense_grid')(y)
    # y = layers.Dense(14, name='dense_grid')(input_grid)
    # grid = keras.Model(input_grid, y)
    # return grid
    return input_grid


def create_combine_network():
    # let's first create a network for each post code
    # 6010, 6014, 6011, 6280, 6281, 6284
    pc_6010 = create_network(6010)
    pc_6014 = create_network(6014)
    pc_6011 = create_network(6011)
    pc_6280 = create_network(6280)
    pc_6281 = create_network(6281)
    pc_6284 = create_network(6284)

    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')
    grid_network = tf.keras.models.load_model('combined_nn_results/refined_models/saved_models/grid_model')
    grid_network.trainable = False
    y = grid_network(input_grid, training=False)

    #
    # combinedInput = layers.concatenate(
    #     [grid_network.output, pc_6010.output, pc_6014.output, pc_6011.output, pc_6280.output, pc_6281.output,
    #      pc_6284.output])
    combinedInput = layers.concatenate(
        [pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
         pc_6284])
    x = layers.Conv1D(kernel_size=4, filters=32)(combinedInput)
    x = layers.Conv1D(kernel_size=4, filters=32)(x)
    x = layers.MaxPooling1D(padding='same')(x)
    x = layers.Flatten(name='flatten_pc')(x)
    x = layers.Dense(14, activation='linear')(x)

    concat_output = layers.concatenate([y, x])
    final_out = layers.Dense(14, activation='linear')(concat_output)



    # combinedInput = layers.concatenate(
    #     [grid_network, pc_6010.output, pc_6014.output, pc_6011.output, pc_6280.output, pc_6281.output,
    #      pc_6284.output])

    # combinedInput = grid_network
    # x = layers.Conv1D(kernel_size=4, filters=32)(combinedInput)
    # x = layers.MaxPooling1D(padding='same')(x)
    # x = layers.Conv1D(kernel_size=4, filters=32)(x)
    # x = layers.MaxPooling1D(padding='same')(x)
    # x = layers.Conv1D(kernel_size=4, filters=32)(x)
    # x = layers.MaxPooling1D(padding='same')(x)
    # x = layers.LayerNormalization()(combinedInput)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=1, activation='relu')(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=2, activation='relu')(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=4)(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=8)(x)
    # x = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=16)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)

    # cnn_layer = 6
    # dilation_rate = 2
    # dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    # padding = 'causal'
    # use_skip_connections = False
    # return_sequences = True
    # dropout_rate = 0.05
    # name = 'tcn'
    # kernel_initializer = 'he_normal'
    # activation = 'relu'
    # opt = 'adam'
    # use_batch_norm = False
    # use_layer_norm = False
    # use_weight_norm = True
    #
    # x = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates, padding=padding,
    #             use_skip_connections=use_skip_connections, dropout_rate=dropout_rate, return_sequences=return_sequences,
    #             activation=activation, kernel_initializer=kernel_initializer, use_batch_norm=use_batch_norm,
    #             use_layer_norm=use_layer_norm,
    #             use_weight_norm=use_weight_norm, name=name)(x)
    # x = layers.Flatten(name='flatten_combined')(x)
    # x = layers.Dense(14, activation='linear')(x)
    # hf_model = keras.Model(
    #     inputs=[grid_network.input, pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
    #             pc_6284.input], outputs=x)
    hf_model = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=final_out)
    # hf_model = keras.Model(
    #     inputs=[pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
    #             pc_6284], outputs=x)

    # hf_model = keras.Model(
    #     inputs=grid_network, outputs=x)

    hf_model.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam(0.0001),
                     metrics=[tf.metrics.MeanAbsoluteError()])
    return hf_model
