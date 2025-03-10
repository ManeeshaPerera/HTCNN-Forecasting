import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn
from constants import SWIS_POSTCODES, UNIQUE_WEATHER_PCS, Clusters, PC_CLUSTER_MAP


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
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=0.05,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=False,
                          use_layer_norm=False,
                          use_weight_norm=True, name='pc_TCN')(concatenation_pc)
    flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    # dense_1 = layers.Dense(100, activation='linear', name="dense_1")(flatten_pc)
    full_connected_layer_pc = layers.Dense(18, activation='linear', name="prediction_layer_pc")(flatten_pc)

    # gird convolution
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    cnn_layer_grid = 4
    dilation_rates_grid = [dilation_rate ** i for i in range(cnn_layer_grid)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rates_grid,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    # dense_2 = layers.Dense(100, activation='linear', name="dense_grid")(flatten_grid)
    full_connected_layer_grid = layers.Dense(18, activation='linear', name="prediction_layer_grid")(flatten_grid)

    concatenation = layers.concatenate([full_connected_layer_pc, full_connected_layer_grid])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_input)
    swis_pc_grid_parallel_model = keras.Model(inputs=input_layers_pc, outputs=prediction_layer)

    swis_pc_grid_parallel_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                        metrics=[tf.metrics.MeanAbsoluteError()])
    return swis_pc_grid_parallel_model


def pc_2d_conv_with_grid_tcn():
    pc_input = keras.Input(shape=(18 * 1, 1 * 101, 14), name='input_pc')

    conv_2d_layer1 = layers.Conv2D(32, kernel_size=(2, 3), activation='relu')(pc_input)
    conv_2d_layer2 = layers.Conv2D(32, kernel_size=(2, 3), activation='relu')(conv_2d_layer1)
    flatten_out_pc = layers.Flatten(name='flatten_pc')(conv_2d_layer2)
    prediction_layer_pc = layers.Dense(18, activation='linear', name='prediction_pcs')(flatten_out_pc)

    # gird convolution
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    cnn_layer_grid = 4
    dilation_rates_grid = [2 ** i for i in range(cnn_layer_grid)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rates_grid,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    grid_prediction_layer = layers.Dense(18, activation='linear', name='prediction_grid')(flatten_grid)

    concatenation = layers.concatenate([prediction_layer_pc, grid_prediction_layer])
    # dense_layer_1 = layers.Dense(100, activation='linear', name="dense_layer1")(concatenation)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    pc_2d_conv_with_grid_tcn_model = keras.Model(
        inputs=[pc_input, grid_input], outputs=prediction_layer)

    pc_2d_conv_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                           metrics=[tf.metrics.MeanAbsoluteError()])
    return pc_2d_conv_with_grid_tcn_model


def pc_2d_conv_with_grid_tcn_method2():
    pc_input = keras.Input(shape=(18 * 1, 1 * 101, 14), name='input_pc')

    conv_2d_layer1 = layers.Conv2D(32, kernel_size=(2, 3), activation='relu')(pc_input)
    conv_2d_layer2 = layers.Conv2D(32, kernel_size=(2, 3), activation='relu')(conv_2d_layer1)
    max_pool_stage = layers.MaxPooling2D()(conv_2d_layer2)
    conv_2d_layer3 = layers.Conv2D(32, kernel_size=(2, 3), activation='relu')(max_pool_stage)
    conv_2d_layer4 = layers.Conv2D(32, kernel_size=(2, 3), activation='relu')(conv_2d_layer3)
    max_pool_stage2 = layers.MaxPooling2D()(conv_2d_layer4)
    flatten_out_pc = layers.Flatten(name='flatten_pc')(max_pool_stage2)
    dense_layer_1 = layers.Dense(100, activation='linear', name="dense_layer1")(flatten_out_pc)
    prediction_layer_pc = layers.Dense(18, activation='linear', name='prediction_pcs')(dense_layer_1)

    # gird convolution
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    cnn_layer_grid = 6
    dilation_rates_grid = [2 ** i for i in range(cnn_layer_grid)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rates_grid,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=True,
                       use_weight_norm=False, name='TCN_grid')(grid_input)
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    grid_prediction_layer = layers.Dense(18, activation='linear', name='prediction_grid')(flatten_grid)

    concatenation = layers.concatenate([prediction_layer_pc, grid_prediction_layer])
    # dense_layer_1 = layers.Dense(100, activation='linear', name="dense_layer1")(concatenation)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    pc_2d_conv_with_grid_tcn_model = keras.Model(
        inputs=[pc_input, grid_input], outputs=prediction_layer)

    pc_2d_conv_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                           metrics=[tf.metrics.MeanAbsoluteError()])
    return pc_2d_conv_with_grid_tcn_model


def grid_conv_in_each_pc_seperately(ts):
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')

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

    # pass the grid input with Convolution
    cnn_layer = 4
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rates,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)

    tcn_pc_output = local_convolution_TCN(input_layer, tcn_grid, ts)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_pc_output)
    full_connected_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    grid_conv_in_each_pc_seperately_model = keras.Model(inputs=[input_layer, grid_input], outputs=full_connected_layer)

    grid_conv_in_each_pc_seperately_model.compile(loss=tf.losses.MeanSquaredError(),
                                                  optimizer=tf.optimizers.Adam(0.0001),
                                                  metrics=[tf.metrics.MeanAbsoluteError()])
    return grid_conv_in_each_pc_seperately_model


def grid_only_network_SWIS_SKIP():
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
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
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    full_connected_layer = layers.Dense(18, activation='linear', name="prediction_layer_grid")(flatten_grid)
    grid_only_network_model = keras.Model(grid_input, full_connected_layer)
    return grid_only_network_model


def SWIS_APPROACH_A_more_layer_without_norm_grid_skip():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    # pc_normalization = layers.LayerNormalization()(concatenation_pc)
    cnn_layer = 10
    tcn_stacks = 6
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    padding = 'causal'
    use_skip_connections = True
    return_sequences = True
    dropout_rate = 0.05
    kernel_initializer = 'he_normal'
    activation = 'relu'
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=tcn_stacks, dilations=dilation_rates,
                          padding=padding,
                          use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                          return_sequences=return_sequences,
                          activation=activation, kernel_initializer=kernel_initializer,
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)
    flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    full_connected_layer_pc = layers.Dense(18, activation='linear', name="prediction_layer_pc")(flatten_pc)

    grid_model = grid_only_network_SWIS_SKIP()

    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_model.input)
    SWIS_APPROACH_A_more_layer_without_norm_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    SWIS_APPROACH_A_more_layer_without_norm_model.compile(loss=tf.losses.MeanSquaredError(),
                                                          optimizer=tf.optimizers.Adam(0.0001),
                                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return SWIS_APPROACH_A_more_layer_without_norm_model


def grid_only_simple():
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    cnn_layer = 4
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rates,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    full_connected_layer = layers.Dense(18, activation='linear', name="prediction_layer_grid")(flatten_grid)
    grid_simple_model = keras.Model(grid_input, full_connected_layer)
    return grid_simple_model


def SWIS_APPROACH_A_with_weather_only():
    input_layers_pc = []
    for ts in UNIQUE_WEATHER_PCS[1:]:
        input_layer = keras.Input(shape=(18 * 1, 7), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    # pc_normalization = layers.LayerNormalization()(concatenation_pc)
    cnn_layer = 6
    tcn_stacks = 2
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    padding = 'causal'
    use_skip_connections = True
    return_sequences = True
    dropout_rate = 0.05
    kernel_initializer = 'he_normal'
    activation = 'relu'
    use_batch_norm = False
    use_layer_norm = True
    use_weight_norm = False
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=tcn_stacks, dilations=dilation_rates,
                          padding=padding,
                          use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                          return_sequences=return_sequences,
                          activation=activation, kernel_initializer=kernel_initializer,
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)
    flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    full_connected_layer_pc = layers.Dense(18, activation='linear', name="prediction_layer_pc")(flatten_pc)

    grid_model = grid_only_simple()
    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_model.input)
    swis_weather_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    swis_weather_model.compile(loss=tf.losses.MeanSquaredError(),
                               optimizer=tf.optimizers.Adam(0.0001),
                               metrics=[tf.metrics.MeanAbsoluteError()])
    return swis_weather_model


def concat_pc_with_grid_tcn2():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn_model


def concat_pc_with_grid_tcn2_with_batchnorm():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = True
    use_layer_norm = False
    use_weight_norm = False
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=True,
                       use_layer_norm=False,
                       use_weight_norm=False, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn2_withoutnorm_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn2_withoutnorm_model.compile(loss=tf.losses.MeanSquaredError(),
                                                       optimizer=tf.optimizers.Adam(0.0001),
                                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn2_withoutnorm_model


def concat_pc_with_grid_tcn2_with_layernorm():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = True
    use_weight_norm = False
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=True,
                       use_weight_norm=False, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn2_withoutlayernorm_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn2_withoutlayernorm_model.compile(loss=tf.losses.MeanSquaredError(),
                                                            optimizer=tf.optimizers.Adam(0.0001),
                                                            metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn2_withoutlayernorm_model


def concat_pc_with_grid_tcn3():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2, 4, 8], 1, concat_each_pc_grid)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer1)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn3_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn3_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                           metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn3_model


def concat_pc_with_grid_tcn2_lr():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn_model


def concat_pc_with_grid_tcn4():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(6)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn_model


def concat_pc_with_grid_tcn4_lr():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(6)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn_model


def concat_pc_with_grid_tcn5():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=False, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn5_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn5_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                           metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn5_model


def concat_pc_with_grid_tcn6():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=6, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=6, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2, 4, 8], 1, concat_each_pc_grid)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer1)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn6_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn6_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                           metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn6_model


def concat_pc_with_grid_at_each_tcn():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, concat_each_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, concat_each_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, concat_each_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn_model


def concat_pc_with_grid_tcn2_new():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    # tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(concat_pc_with_layer3)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn_model


def concat_pc_with_grid_tcn2_relu_and_norm():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    def normalise_output(layer_input):
        batch_norm = layers.BatchNormalization()(layer_input)
        activation = layers.Activation(activation='relu')(batch_norm)
        drop_out = layers.SpatialDropout1D(0.05)(activation)
        return drop_out

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')
    normalise_value1 = normalise_output(concat_each_pc_grid)
    tcn_layer1 = get_tcn_layer([1, 2], 1, normalise_value1)

    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    normalise_value2 = normalise_output(concat_pc_with_layer1)
    tcn_layer2 = get_tcn_layer([1, 4], 2, normalise_value2)

    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    normalise_value3 = normalise_output(concat_pc_with_layer2)
    tcn_layer3 = get_tcn_layer([1, 8], 3, normalise_value3)

    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    normalise_value4 = normalise_output(concat_pc_with_layer3)
    tcn_layer4 = get_tcn_layer([1, 16], 4, normalise_value4)

    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn_model


def concat_pc_with_grid_tcn2_lr_decay():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn2_lr_decay_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    concat_pc_with_grid_tcn2_lr_decay_model.compile(loss=tf.losses.MeanSquaredError(),
                                                    optimizer=tf.optimizers.Adam(lr_schedule),
                                                    metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn2_lr_decay_model


def concat_pc_with_grid_tcn2_concat_at_end():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    concat_pc_with_layer4 = layers.concatenate([tcn_layer4, tcn_pc_grid])
    flatten_out = layers.Flatten(name='flatten_all')(concat_pc_with_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn2_concat_at_end_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn2_concat_at_end_model.compile(loss=tf.losses.MeanSquaredError(),
                                                         optimizer=tf.optimizers.Adam(0.0001),
                                                         metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn2_concat_at_end_model


def concat_pc_with_grid_tcn2_for_cluster(cluster_num):
    input_layers_pc = []
    count =0
    for ts in Clusters[cluster_num]:
        if count == 0:
            input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        else:
            input_layer = keras.Input(shape=(18 * 1, 7), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)
        count=count+1

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn2_for_cluster_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn2_for_cluster_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn2_for_cluster_model


def concat_pc_with_grid_tcn2_weather_centroids():
    input_layers_pc = []
    cluster_map = []
    for ts in SWIS_POSTCODES:
        cluster_id = PC_CLUSTER_MAP[ts]
        if cluster_id not in cluster_map:
            input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
            cluster_map.append(cluster_id)
        else:
            input_layer = keras.Input(shape=(18 * 1, 7), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    dilation_rates = [2 ** i for i in range(4)]
    dropout_rate = 0.05
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=3, dilations=dilation_rates,
                          padding='causal',
                          use_skip_connections=True, dropout_rate=dropout_rate,
                          return_sequences=True,
                          activation='relu', kernel_initializer='he_normal',
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(concatenation_pc)

    # create the grid model
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')

    def get_tcn_layer(dilation_rate, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=True, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_grid')(input_to_layer)

    # dilation_rates = [2 ** i for i in range(4)]

    concat_each_pc_grid = layers.concatenate([tcn_pc_grid, grid_input], name=f'concat_pc_grid')

    tcn_layer1 = get_tcn_layer([1, 2], 1, concat_each_pc_grid)
    concat_pc_with_layer1 = layers.concatenate([tcn_layer1, tcn_pc_grid])
    tcn_layer2 = get_tcn_layer([1, 4], 2, concat_pc_with_layer1)
    concat_pc_with_layer2 = layers.concatenate([tcn_layer2, tcn_pc_grid])
    tcn_layer3 = get_tcn_layer([1, 8], 3, concat_pc_with_layer2)
    concat_pc_with_layer3 = layers.concatenate([tcn_layer3, tcn_pc_grid])
    tcn_layer4 = get_tcn_layer([1, 16], 4, concat_pc_with_layer3)
    flatten_out = layers.Flatten(name='flatten_all')(tcn_layer4)
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(flatten_out)

    input_layers_pc.append(grid_input)
    concat_pc_with_grid_tcn2_weather_centroids_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    concat_pc_with_grid_tcn2_weather_centroids_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return concat_pc_with_grid_tcn2_weather_centroids_model