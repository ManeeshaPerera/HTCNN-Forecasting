import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn
from tensorflow.keras import regularizers


def sequentional_training_approach():
    def get_tcn_layer(dilation_rate, pc, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_{pc}_grid')(input_to_layer)

    def local_convolution_TCN(pc_ts, grid_conv_values, pc):
        cnn_layer = 6
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]

        concat_each_pc_grid = layers.concatenate([pc_ts, grid_conv_values], name=f'concat_{pc}_grid')

        tcn_layer1 = get_tcn_layer([dilation_rates[0]], pc, 1, concat_each_pc_grid)
        concat_grid_with_layer1 = layers.concatenate([tcn_layer1, grid_conv_values])
        tcn_layer2 = get_tcn_layer([dilation_rates[1]], pc, 2, concat_grid_with_layer1)
        concat_grid_with_layer2 = layers.concatenate([tcn_layer2, grid_conv_values])
        tcn_layer3 = get_tcn_layer([dilation_rates[2]], pc, 3, concat_grid_with_layer2)
        concat_grid_with_layer3 = layers.concatenate([tcn_layer3, grid_conv_values])
        tcn_layer4 = get_tcn_layer([dilation_rates[3]], pc, 4, concat_grid_with_layer3)

        return tcn_layer4

    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    pc_input = keras.Input(shape=(18 * 1, 14), name='input_pc')

    # pass the grid input with Convolution
    cnn_layer = 4
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)

    tcn_pc_output = local_convolution_TCN(pc_input, tcn_grid, 'pc')
    lstm_layer1 = layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2())(tcn_pc_output)
    lstm_layer2 = layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2())(lstm_layer1)
    flatten_out = layers.Flatten(name='flatten_all')(lstm_layer2)
    full_connected_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    grid_conv_added_at_each_TCN_together_model = keras.Model(
        inputs=[grid_input, pc_input], outputs=full_connected_layer)

    grid_conv_added_at_each_TCN_together_model.compile(loss=tf.losses.MeanSquaredError(),
                                                       optimizer=tf.optimizers.Adam(0.0001),
                                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return grid_conv_added_at_each_TCN_together_model


def pc_together_2D_conv_approach():
    pc_input = keras.Input(shape=(18 * 1, 1 * 101, 14), name='input_pc')

    conv_2d_layer1 = layers.Conv2D(32, kernel_size=(4, 4))(pc_input)
    conv_2d_layer2 = layers.Conv2D(32, kernel_size=(4, 4))(conv_2d_layer1)
    flatten_out = layers.Flatten(name='flatten_all')(conv_2d_layer2)
    full_connected_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    pc_together_2D_conv_approach_model = keras.Model(
        inputs=[pc_input], outputs=full_connected_layer)

    pc_together_2D_conv_approach_model.compile(loss=tf.losses.MeanSquaredError(),
                                               optimizer=tf.optimizers.Adam(0.0001),
                                               metrics=[tf.metrics.MeanAbsoluteError()])
    return pc_together_2D_conv_approach_model
