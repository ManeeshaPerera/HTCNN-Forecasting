import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn


def grid_conv_added_at_each_TCN_together_skip_connection_True():
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

    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
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

    pc_6010_tcn_out = local_convolution_TCN(pc_6010, tcn_grid, 6010)
    pc_6014_tcn_out = local_convolution_TCN(pc_6014, tcn_grid, 6014)
    pc_6011_tcn_out = local_convolution_TCN(pc_6011, tcn_grid, 6011)
    pc_6280_tcn_out = local_convolution_TCN(pc_6280, tcn_grid, 6280)
    pc_6281_tcn_out = local_convolution_TCN(pc_6281, tcn_grid, 6281)
    pc_6284_tcn_out = local_convolution_TCN(pc_6284, tcn_grid, 6284)

    concat_features = layers.concatenate(
        [pc_6010_tcn_out, pc_6014_tcn_out, pc_6011_tcn_out, pc_6280_tcn_out, pc_6281_tcn_out, pc_6284_tcn_out],
        name='concatenate_all')
    flatten_out = layers.Flatten(name='flatten_all')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    grid_conv_added_at_each_TCN_together_model = keras.Model(
        inputs=[grid_input, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer)

    grid_conv_added_at_each_TCN_together_model.compile(loss=tf.losses.MeanSquaredError(),
                                                       optimizer=tf.optimizers.Adam(0.0001),
                                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return grid_conv_added_at_each_TCN_together_model


def grid_conv_added_at_each_CNN_together():
    def get_cnn_layer(pc, layer_num, input_to_layer):
        cnn_layer1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_{pc}_layer1_{layer_num}')(
            input_to_layer)
        cnn_layer2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_{pc}_layer2_{layer_num}')(
            cnn_layer1)
        max_pool_stage = layers.MaxPooling1D(padding='same', strides=1)(cnn_layer2)
        return max_pool_stage

    def local_convolution_TCN(pc_ts, grid_conv_values, pc):
        concat_each_pc_grid = layers.concatenate([pc_ts, grid_conv_values], name=f'concat_{pc}_grid')

        global_cnn_layer1 = get_cnn_layer(pc, 1, concat_each_pc_grid)
        concat_grid_with_layer1 = layers.concatenate([global_cnn_layer1, grid_conv_values])
        global_cnn_layer2 = get_cnn_layer(pc, 2, concat_grid_with_layer1)
        concat_grid_with_layer2 = layers.concatenate([global_cnn_layer2, grid_conv_values])
        global_cnn_layer3 = get_cnn_layer(pc, 3, concat_grid_with_layer2)
        concat_grid_with_layer3 = layers.concatenate([global_cnn_layer3, grid_conv_values])
        global_cnn_layer4 = get_cnn_layer(pc, 4, concat_grid_with_layer3)

        return global_cnn_layer4

    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
    # pass the grid input with Convolution
    cnn_layer1_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer1_grid')(grid_input)
    cnn_layer2_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer2_grid')(cnn_layer1_grid)
    max_pool_stage = layers.MaxPooling1D(padding='same', strides=1)(cnn_layer2_grid)
    cnn_layer3_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer3_grid')(max_pool_stage)
    cnn_layer4_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer4_grid')(cnn_layer3_grid)
    max_pool_stage_2 = layers.MaxPooling1D(padding='same', strides=1)(cnn_layer4_grid)

    pc_6010_cnn_out = local_convolution_TCN(pc_6010, max_pool_stage_2, 6010)
    pc_6014_cnn_out = local_convolution_TCN(pc_6014, max_pool_stage_2, 6014)
    pc_6011_cnn_out = local_convolution_TCN(pc_6011, max_pool_stage_2, 6011)
    pc_6280_cnn_out = local_convolution_TCN(pc_6280, max_pool_stage_2, 6280)
    pc_6281_cnn_out = local_convolution_TCN(pc_6281, max_pool_stage_2, 6281)
    pc_6284_cnn_out = local_convolution_TCN(pc_6284, max_pool_stage_2, 6284)

    concat_features = layers.concatenate(
        [pc_6010_cnn_out, pc_6014_cnn_out, pc_6011_cnn_out, pc_6280_cnn_out, pc_6281_cnn_out, pc_6284_cnn_out],
        name='concatenate_all')
    flatten_out = layers.Flatten(name='flatten_all')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    grid_conv_added_at_each_cnn_together_model = keras.Model(
        inputs=[grid_input, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer)

    grid_conv_added_at_each_cnn_together_model.compile(loss=tf.losses.MeanSquaredError(),
                                                       optimizer=tf.optimizers.Adam(0.0001),
                                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return grid_conv_added_at_each_cnn_together_model


def possibility_a_postcode_only_separate_paths():
    def get_tcn_layer(dilation_rate, pc, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_{pc}_grid')(input_to_layer)

    def local_convolution_TCN(pc_ts, pc):
        cnn_layer = 4
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]

        tcn_layer1 = get_tcn_layer([dilation_rates[0]], pc, 1, pc_ts)
        tcn_layer2 = get_tcn_layer([dilation_rates[1]], pc, 2, tcn_layer1)
        tcn_layer3 = get_tcn_layer([dilation_rates[2]], pc, 3, tcn_layer2)
        tcn_layer4 = get_tcn_layer([dilation_rates[3]], pc, 4, tcn_layer3)

        return tcn_layer4

    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    pc_6010_tcn_out = local_convolution_TCN(pc_6010, 6010)
    pc_6014_tcn_out = local_convolution_TCN(pc_6014, 6014)
    pc_6011_tcn_out = local_convolution_TCN(pc_6011, 6011)
    pc_6280_tcn_out = local_convolution_TCN(pc_6280, 6280)
    pc_6281_tcn_out = local_convolution_TCN(pc_6281, 6281)
    pc_6284_tcn_out = local_convolution_TCN(pc_6284, 6284)

    concat_features = layers.concatenate(
        [pc_6010_tcn_out, pc_6014_tcn_out, pc_6011_tcn_out, pc_6280_tcn_out, pc_6281_tcn_out, pc_6284_tcn_out],
        name='concatenate_all')
    flatten_out = layers.Flatten(name='flatten_all')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    postcode_only_separate_paths_model = keras.Model(
        inputs=[pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer)

    postcode_only_separate_paths_model.compile(loss=tf.losses.MeanSquaredError(),
                                                       optimizer=tf.optimizers.Adam(0.0001),
                                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return postcode_only_separate_paths_model

def grid_level_branch():
    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
    # pass the grid input with Convolution
    cnn_layer = 4
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer_grid")(flatten_grid)
    grid_only_network_model = keras.Model(grid_input, full_connected_layer)
    grid_only_network_model.compile(loss=tf.losses.MeanSquaredError(),
                                    optimizer=tf.optimizers.Adam(0.0001),
                                    metrics=[tf.metrics.MeanAbsoluteError()])

    return grid_only_network_model

def postcode_level_branch(pc):
    def get_tcn_layer(dilation_rate, pc, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
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


    pc_data = keras.Input(shape=(14 * 1, 14), name=f'input_postcode_{pc}')

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
    # pass the grid input with Convolution
    cnn_layer = 4
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)

    pc_tcn_out = local_convolution_TCN(pc_data, tcn_grid, pc)
    flatten_out = layers.Flatten(name='flatten_all')(pc_tcn_out)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    postcode_level_branch_approach = keras.Model(
        inputs=[grid_input, pc_data], outputs=full_connected_layer)

    postcode_level_branch_approach.compile(loss=tf.losses.MeanSquaredError(),
                                                       optimizer=tf.optimizers.Adam(0.0001),
                                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return postcode_level_branch_approach
