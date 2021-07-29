import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn
from constants import SWIS_POSTCODES


def swis_parallel_ts():
    # construction of input layers
    input_layers = []
    ts_names = []
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    input_layers.append(grid_input)
    ts_names.append('grid')

    for pc in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{pc}')
        input_layers.append(input_layer)
        ts_names.append(str(pc))

    def local_convolution_ts(ts, input_ts):
        cnn_layer = 4
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
        tcn_ts_output = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                                padding='causal',
                                use_skip_connections=True, dropout_rate=0.05,
                                return_sequences=True,
                                activation='relu', kernel_initializer='he_normal',
                                use_batch_norm=False,
                                use_layer_norm=False,
                                use_weight_norm=True, name=f'TCN_{ts}')(input_ts)
        flatten_ts = layers.Flatten(name=f'flatten_{ts}')(tcn_ts_output)
        return flatten_ts

    all_tcn_outputs = []
    for ts_index in range(0, len(ts_names)):
        ts_flatten = local_convolution_ts(ts_names[ts_index], input_layers[ts_index])
        all_tcn_outputs.append(ts_flatten)

    concat_layer = layers.concatenate(all_tcn_outputs)
    dense_1 = layers.Dense(100, activation='linear', name="dense_1")(concat_layer)
    final_fully_connect_layer = layers.Dense(18, activation='linear', name="prediction_layer")(dense_1)

    swis_parallel_ts_model = keras.Model(inputs=input_layers, outputs=final_fully_connect_layer)

    swis_parallel_ts_model.compile(loss=tf.losses.MeanSquaredError(),
                                   optimizer=tf.optimizers.Adam(0.0001),
                                   metrics=[tf.metrics.MeanAbsoluteError()])
    return swis_parallel_ts_model


def swis_pc_grid_parallel():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc, name='postcode_concat')
    cnn_layer = 10
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=6, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=0.05,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=False,
                          use_layer_norm=False,
                          use_weight_norm=True, name='pc_TCN')(concatenation_pc)
    flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)

    # gird convolution
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    cnn_layer_grid = 4
    dilation_rates_grid = [dilation_rate ** i for i in range(cnn_layer_grid)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer_grid, dilations=dilation_rates_grid,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)

    concatenation = layers.concatenate([flatten_grid, flatten_pc])
    dense_1 = layers.Dense(30, activation='linear', name="dense_1")(concatenation)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(dense_1)

    input_layers_pc.append(grid_input)
    swis_pc_grid_parallel_model = keras.Model(inputs=input_layers_pc, outputs=prediction_layer)

    swis_pc_grid_parallel_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                        metrics=[tf.metrics.MeanAbsoluteError()])
    return swis_pc_grid_parallel_model


def pc_2d_conv_with_grid_tcn():
    pc_input = keras.Input(shape=(18 * 1, 1 * 101, 14), name='input_pc')

    conv_2d_layer1 = layers.Conv2D(32, kernel_size=(4, 4), activation='relu')(pc_input)
    conv_2d_layer2 = layers.Conv2D(32, kernel_size=(4, 4), activation='relu')(conv_2d_layer1)
    flatten_out_pc = layers.Flatten(name='flatten_pc')(conv_2d_layer2)

    # gird convolution
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    cnn_layer_grid = 4
    dilation_rates_grid = [2 ** i for i in range(cnn_layer_grid)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer_grid, dilations=dilation_rates_grid,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)

    concatenation = layers.concatenate([flatten_grid, flatten_out_pc])
    dense_layer_1 = layers.Dense(100, activation='linear', name="dense_layer1")(concatenation)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(dense_layer_1)

    pc_2d_conv_with_grid_tcn_model = keras.Model(
        inputs=[pc_input, grid_input], outputs=prediction_layer)

    pc_2d_conv_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0005),
                                           metrics=[tf.metrics.MeanAbsoluteError()])
    return pc_2d_conv_with_grid_tcn_model
