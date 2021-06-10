import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn

# LOCAL AND FULL CONVOLUTION PIPELINE

def local_convolution_stage(pc):
    input_postcode = keras.Input(shape=(14 * 1, 14), name=f'input_postcode_{pc}')
    conv1 = layers.Conv1D(kernel_size=2, padding='same', filters=32, name=f'conv1_postcode_{pc}', activation='relu')(
        input_postcode)
    max_pool = layers.MaxPooling1D(padding='same', strides=1)(conv1)
    model_pc = keras.Model(input_postcode, max_pool)
    return model_pc


def local_and_full_convolution_approach():
    pc_6010 = local_convolution_stage(6010)
    pc_6014 = local_convolution_stage(6014)
    pc_6011 = local_convolution_stage(6011)
    pc_6280 = local_convolution_stage(6280)
    pc_6281 = local_convolution_stage(6281)
    pc_6284 = local_convolution_stage(6284)

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
    concatenation = layers.concatenate(
        [grid_input, pc_6010.output, pc_6014.output, pc_6011.output, pc_6280.output, pc_6281.output,
         pc_6284.output])
    conv_full_stage = layers.Conv1D(kernel_size=4, filters=32, name='conv1_full_stage')(concatenation)
    conv2_full_stage = layers.Conv1D(kernel_size=4, filters=32, name='conv2_full_stage')(conv_full_stage)
    max_pool_full_stage = layers.MaxPooling1D(padding='same', name='max_pool_full_stage')(conv2_full_stage)
    flatten_out = layers.Flatten(name='flatten_full_stage')(max_pool_full_stage)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    local_and_full_conv_model = keras.Model(
        inputs=[grid_input, pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
                pc_6284.input], outputs=full_connected_layer)

    local_and_full_conv_model.compile(loss=tf.losses.MeanSquaredError(),
                                      optimizer=tf.optimizers.Adam(0.0001),
                                      metrics=[tf.metrics.MeanAbsoluteError()])
    return local_and_full_conv_model


def local_and_full_convolution_approach_alternative1():
    # removing the Grid level Branch
    pc_6010 = local_convolution_stage(6010)
    pc_6014 = local_convolution_stage(6014)
    pc_6011 = local_convolution_stage(6011)
    pc_6280 = local_convolution_stage(6280)
    pc_6281 = local_convolution_stage(6281)
    pc_6284 = local_convolution_stage(6284)

    concatenation = layers.concatenate(
        [pc_6010.output, pc_6014.output, pc_6011.output, pc_6280.output, pc_6281.output,
         pc_6284.output])
    conv_full_stage = layers.Conv1D(kernel_size=4, filters=32, name='conv1_full_stage')(concatenation)
    conv2_full_stage = layers.Conv1D(kernel_size=4, filters=32, name='conv2_full_stage')(conv_full_stage)
    max_pool_full_stage = layers.MaxPooling1D(padding='same', name='max_pool_full_stage')(conv2_full_stage)
    flatten_out = layers.Flatten(name='flatten_full_stage')(max_pool_full_stage)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    local_and_full_conv_model = keras.Model(
        inputs=[pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
                pc_6284.input], outputs=full_connected_layer)

    local_and_full_conv_model.compile(loss=tf.losses.MeanSquaredError(),
                                      optimizer=tf.optimizers.Adam(0.0001),
                                      metrics=[tf.metrics.MeanAbsoluteError()])
    return local_and_full_conv_model


def local_and_full_convolution_approach_alternative2():
    # removing Grid level and Local Convolution Branches
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    concatenation = layers.concatenate(
        [pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
         pc_6284])
    conv_full_stage = layers.Conv1D(kernel_size=4, filters=32, name='conv1_full_stage')(concatenation)
    conv2_full_stage = layers.Conv1D(kernel_size=4, filters=32, name='conv2_full_stage')(conv_full_stage)
    max_pool_full_stage = layers.MaxPooling1D(padding='same', name='max_pool_full_stage')(conv2_full_stage)
    flatten_out = layers.Flatten(name='flatten_full_stage')(max_pool_full_stage)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    local_and_full_conv_model = keras.Model(
        inputs=[pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer)

    local_and_full_conv_model.compile(loss=tf.losses.MeanSquaredError(),
                                      optimizer=tf.optimizers.Adam(0.0001),
                                      metrics=[tf.metrics.MeanAbsoluteError()])
    return local_and_full_conv_model


# FROZEN BRANCH APPROACH

def frozen_branch_approach():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    concatenation = layers.concatenate(
        [pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
         pc_6284])
    pc_conv = layers.Conv1D(kernel_size=4, filters=32, padding='same', name='conv1_pc')(concatenation)
    pc_conv2 = layers.Conv1D(kernel_size=4, filters=32, padding='same', name='conv2_pc')(pc_conv)
    flatten_pc = layers.Flatten(name='flatten_pc')(pc_conv2)
    full_connected_layer_pc = layers.Dense(14, activation='linear', name="prediction_layer_pc")(flatten_pc)

    # LOAD PRETRAINED GRID MODEL
    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')
    grid_network = tf.keras.models.load_model('combined_nn_results/refined_models/saved_models/grid_model')
    grid_network.trainable = False
    grid_model = grid_network(input_grid, training=False)

    concatenation = layers.concatenate([grid_model, full_connected_layer_pc])
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(concatenation)

    frozen_branch_model = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    frozen_branch_model.compile(loss=tf.losses.MeanSquaredError(),
                                optimizer=tf.optimizers.Adam(0.0001),
                                metrics=[tf.metrics.MeanAbsoluteError()])
    return frozen_branch_model


def last_residual_approach():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')

    # postcode convolutions
    concatenation_pc = layers.concatenate(
        [pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
         pc_6284], name='postcode_concat')
    pc_normalization = layers.LayerNormalization()(concatenation_pc)
    pc_conv1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=1, name="pc_conv1")(
        pc_normalization)
    pc_conv2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=2, name="pc_conv2")(pc_conv1)

    # Grid convolutions
    grid_conv1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=1, name="grid_conv1")(
        input_grid)
    grid_conv2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=2, name="grid_conv2")(
        grid_conv1)

    # concatenation
    concat_grid_conv_pc_conv = layers.concatenate([grid_conv2, pc_conv2], name='grid_pc_concat')
    concat_grid_conv_pc_conv_normal = layers.LayerNormalization()(concat_grid_conv_pc_conv)

    # Convolution
    pc_grid_conv1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, dilation_rate=1, name="grid_pc_conv1")(
        concat_grid_conv_pc_conv_normal)
    pc_grid_conv2 = layers.Conv1D(kernel_size=2, padding='causal', filters=64, dilation_rate=1, name="grid_pc_conv2")(
        pc_grid_conv1)

    # Skip connection
    skip_connection = layers.add([concat_grid_conv_pc_conv_normal, pc_grid_conv2], name='skip_connection_addition')

    # Fully Connected Layer
    flatten_layer = layers.Flatten(name='flatten_all')(skip_connection)
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_layer)

    last_residual_model = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    last_residual_model.compile(loss=tf.losses.MeanSquaredError(),
                                optimizer=tf.optimizers.Adam(0.0001),
                                metrics=[tf.metrics.MeanAbsoluteError()])
    return last_residual_model


# LOCAL CONV AND ADDING GRID TIME SERIES
def local_conv_with_grid_approach():
    pc_6010 = local_convolution_stage(6010)
    pc_6014 = local_convolution_stage(6014)
    pc_6011 = local_convolution_stage(6011)
    pc_6280 = local_convolution_stage(6280)
    pc_6281 = local_convolution_stage(6281)
    pc_6284 = local_convolution_stage(6284)

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')

    def concat_with_grid(pc_output, grid_in, pc):
        concat_with_local_grid = layers.concatenate([pc_output, grid_in], name=f'concat_grid_{pc}')
        normalize_layer = layers.LayerNormalization()(concat_with_local_grid)
        conv_full_stage = layers.Conv1D(kernel_size=2, filters=32, name=f'conv1_{pc}_grid', activation='relu')(
            normalize_layer)
        conv2_full_stage = layers.Conv1D(kernel_size=2, filters=32, name=f'conv2_{pc}_grid', activation='relu')(
            conv_full_stage)
        max_pool_full_stage = layers.MaxPooling1D(padding='same', name=f'max_pool_{pc}_grid')(conv2_full_stage)
        return max_pool_full_stage

    pc_6010_features = concat_with_grid(pc_6010.output, grid_input, 6010)
    pc_6014_features = concat_with_grid(pc_6014.output, grid_input, 6014)
    pc_6011_features = concat_with_grid(pc_6011.output, grid_input, 6011)
    pc_6280_features = concat_with_grid(pc_6280.output, grid_input, 6280)
    pc_6281_features = concat_with_grid(pc_6281.output, grid_input, 6281)
    pc_6284_features = concat_with_grid(pc_6284.output, grid_input, 6284)

    concat_features = layers.concatenate(
        [pc_6010_features, pc_6014_features, pc_6011_features, pc_6280_features, pc_6281_features, pc_6284_features],
        name='concatenate_all')
    normalize_concat = layers.LayerNormalization()(concat_features)
    flatten_out = layers.Flatten(name='flatten_all')(normalize_concat)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    local_conv_with_grid_model = keras.Model(
        inputs=[grid_input, pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
                pc_6284.input], outputs=full_connected_layer)

    local_conv_with_grid_model.compile(loss=tf.losses.MeanSquaredError(),
                                       optimizer=tf.optimizers.Adam(0.0001),
                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return local_conv_with_grid_model


# LOCAL CONV AND ADDING GRID TIME SERIES WITH TCN
def local_conv_with_grid_with_TCN_approach():
    def local_convolution_TCN(pc):
        cnn_layer = 4
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
        padding = 'causal'
        use_skip_connections = True
        return_sequences = True
        dropout_rate = 0.05
        name = f'TCN_{pc}'
        kernel_initializer = 'he_normal'
        activation = 'relu'
        use_batch_norm = False
        use_layer_norm = False
        use_weight_norm = True

        input_postcode = keras.Input(shape=(14 * 1, 14), name=f'input_postcode_{pc}')
        tcn_pc_output = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                                padding=padding,
                                use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                                return_sequences=return_sequences,
                                activation=activation, kernel_initializer=kernel_initializer,
                                use_batch_norm=use_batch_norm,
                                use_layer_norm=use_layer_norm,
                                use_weight_norm=use_weight_norm, name=name)(input_postcode)
        tcn_pc_model = keras.Model(input_postcode, tcn_pc_output)
        return tcn_pc_model

    pc_6010 = local_convolution_TCN(6010)
    pc_6014 = local_convolution_TCN(6014)
    pc_6011 = local_convolution_TCN(6011)
    pc_6280 = local_convolution_TCN(6280)
    pc_6281 = local_convolution_TCN(6281)
    pc_6284 = local_convolution_TCN(6284)

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')

    def concat_with_grid(pc_output, grid_in, pc):
        concat_with_local_grid = layers.concatenate([pc_output, grid_in], name=f'concat_grid_{pc}')
        normalize_layer = layers.LayerNormalization()(concat_with_local_grid)
        cnn_layer = 6
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
        padding = 'causal'
        use_skip_connections = False
        return_sequences = True
        dropout_rate = 0.05
        name = f'TCN_{pc}_grid'
        kernel_initializer = 'he_normal'
        activation = 'relu'
        use_batch_norm = False
        use_layer_norm = False
        use_weight_norm = True
        tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                              padding=padding,
                              use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                              return_sequences=return_sequences,
                              activation=activation, kernel_initializer=kernel_initializer,
                              use_batch_norm=use_batch_norm,
                              use_layer_norm=use_layer_norm,
                              use_weight_norm=use_weight_norm, name=name)(normalize_layer)
        return tcn_pc_grid

    pc_6010_features = concat_with_grid(pc_6010.output, grid_input, 6010)
    pc_6014_features = concat_with_grid(pc_6014.output, grid_input, 6014)
    pc_6011_features = concat_with_grid(pc_6011.output, grid_input, 6011)
    pc_6280_features = concat_with_grid(pc_6280.output, grid_input, 6280)
    pc_6281_features = concat_with_grid(pc_6281.output, grid_input, 6281)
    pc_6284_features = concat_with_grid(pc_6284.output, grid_input, 6284)

    concat_features = layers.concatenate(
        [pc_6010_features, pc_6014_features, pc_6011_features, pc_6280_features, pc_6281_features, pc_6284_features],
        name='concatenate_all')
    normalize_concat = layers.LayerNormalization()(concat_features)
    flatten_out = layers.Flatten(name='flatten_all')(normalize_concat)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    local_conv_with_grid_model = keras.Model(
        inputs=[grid_input, pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
                pc_6284.input], outputs=full_connected_layer)

    local_conv_with_grid_model.compile(loss=tf.losses.MeanSquaredError(),
                                       optimizer=tf.optimizers.Adam(0.0001),
                                       metrics=[tf.metrics.MeanAbsoluteError()])
    return local_conv_with_grid_model


def last_residual_approach_with_TCN():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')

    # postcode convolutions
    concatenation_pc = layers.concatenate([pc_6010, pc_6014, pc_6011, pc_6280, pc_6281, pc_6284],
                                          name='postcode_concat')
    pc_normalization = layers.LayerNormalization()(concatenation_pc)
    cnn_layer = 6
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    padding = 'causal'
    use_skip_connections = False
    return_sequences = True
    dropout_rate = 0.05
    kernel_initializer = 'he_normal'
    activation = 'relu'
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                          padding=padding,
                          use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                          return_sequences=return_sequences,
                          activation=activation, kernel_initializer=kernel_initializer,
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(pc_normalization)

    # concatenation
    concat_grid_conv_pc_conv = layers.concatenate([input_grid, tcn_pc_grid], name='grid_pcTCN_concat')
    concat_grid_conv_pc_conv_normal = layers.LayerNormalization()(concat_grid_conv_pc_conv)

    # Convolution
    cnn_layer_full = 6
    dilation_rate_full = 2
    dilation_rates_full = [dilation_rate_full ** i for i in range(cnn_layer_full)]
    tcn_full = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer_full, dilations=dilation_rates_full,
                       padding=padding,
                       use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                       return_sequences=return_sequences,
                       activation=activation, kernel_initializer=kernel_initializer,
                       use_batch_norm=use_batch_norm,
                       use_layer_norm=use_layer_norm,
                       use_weight_norm=use_weight_norm, name='full_TCN')(concat_grid_conv_pc_conv_normal)

    # Skip connection
    skip_connection = layers.add([tcn_pc_grid, tcn_full], name='skip_connection_addition')

    # Fully Connected Layer
    flatten_layer = layers.Flatten(name='flatten_all')(skip_connection)
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_layer)

    last_residual_model_TCN = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    last_residual_model_TCN.compile(loss=tf.losses.MeanSquaredError(),
                                    optimizer=tf.optimizers.Adam(0.0001),
                                    metrics=[tf.metrics.MeanAbsoluteError()])
    return last_residual_model_TCN


def postcode_only_TCN():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    concatenation_pc = layers.concatenate([pc_6010, pc_6014, pc_6011, pc_6280, pc_6281, pc_6284],
                                          name='postcode_concat')

    # postcode convolutions
    cnn_layer = 6
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]

    tcn_pc = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                     padding='causal',
                     use_skip_connections=False, dropout_rate=0.05,
                     return_sequences=True,
                     activation='relu', kernel_initializer='he_normal',
                     use_batch_norm=False,
                     use_layer_norm=False,
                     use_weight_norm=True, name='TCN_PC')(concatenation_pc)

    flatten_layer = layers.Flatten(name='flatten_tcn_features')(tcn_pc)
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_layer)

    postcode_only_tcn_model = keras.Model(
        inputs=[pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    postcode_only_tcn_model.compile(loss=tf.losses.MeanSquaredError(),
                                    optimizer=tf.optimizers.Adam(0.0001),
                                    metrics=[tf.metrics.MeanAbsoluteError()])
    return postcode_only_tcn_model


# LOCAL CONV AND ADDING GRID TIME SERIES WITH CONV
def local_conv_with_grid_conv_TCN_approach():
    def local_convolution_TCN(pc):
        cnn_layer = 4
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]

        input_postcode = keras.Input(shape=(14 * 1, 14), name=f'input_postcode_{pc}')
        tcn_pc_output = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                                padding='causal',
                                use_skip_connections=False, dropout_rate=0.05,
                                return_sequences=True,
                                activation='relu', kernel_initializer='he_normal',
                                use_batch_norm=False,
                                use_layer_norm=False,
                                use_weight_norm=True, name=f'TCN_{pc}')(input_postcode)
        tcn_pc_model = keras.Model(input_postcode, tcn_pc_output)
        return tcn_pc_model

    pc_6010 = local_convolution_TCN(6010)
    pc_6014 = local_convolution_TCN(6014)
    pc_6011 = local_convolution_TCN(6011)
    pc_6280 = local_convolution_TCN(6280)
    pc_6281 = local_convolution_TCN(6281)
    pc_6284 = local_convolution_TCN(6284)

    # pass the grid input with Convolution
    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
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

    def concat_with_grid(pc_output, grid_conv_output, pc):
        concat_with_local_grid = layers.concatenate([pc_output, grid_conv_output], name=f'concat_grid_{pc}')
        cnn_layer = 6
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
        tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                              padding='causal',
                              use_skip_connections=False, dropout_rate=0.05,
                              return_sequences=True,
                              activation='relu', kernel_initializer='he_normal',
                              use_batch_norm=False,
                              use_layer_norm=False,
                              use_weight_norm=True, name=f'TCN_{pc}_grid')(concat_with_local_grid)
        return tcn_pc_grid

    pc_6010_features = concat_with_grid(pc_6010.output, tcn_grid, 6010)
    pc_6014_features = concat_with_grid(pc_6014.output, tcn_grid, 6014)
    pc_6011_features = concat_with_grid(pc_6011.output, tcn_grid, 6011)
    pc_6280_features = concat_with_grid(pc_6280.output, tcn_grid, 6280)
    pc_6281_features = concat_with_grid(pc_6281.output, tcn_grid, 6281)
    pc_6284_features = concat_with_grid(pc_6284.output, tcn_grid, 6284)

    concat_features = layers.concatenate(
        [pc_6010_features, pc_6014_features, pc_6011_features, pc_6280_features, pc_6281_features, pc_6284_features],
        name='concatenate_all')
    flatten_out = layers.Flatten(name='flatten_all')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    local_conv_with_grid_conv_TCN_model = keras.Model(
        inputs=[grid_input, pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
                pc_6284.input], outputs=full_connected_layer)

    local_conv_with_grid_conv_TCN_model.compile(loss=tf.losses.MeanSquaredError(),
                                                optimizer=tf.optimizers.Adam(0.0001),
                                                metrics=[tf.metrics.MeanAbsoluteError()])
    return local_conv_with_grid_conv_TCN_model


def pc_and_grid_input_together():
    def local_convolution_TCN(pc_ts, grid_ts, pc):
        cnn_layer = 6
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
        concat_each_pc_grid = layers.concatenate([pc_ts, grid_ts], name=f'concat_{pc}_grid')

        tcn_pc_grid_output = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                                     padding='causal',
                                     use_skip_connections=False, dropout_rate=0.05,
                                     return_sequences=True,
                                     activation='relu', kernel_initializer='he_normal',
                                     use_batch_norm=False,
                                     use_layer_norm=False,
                                     use_weight_norm=True, name=f'TCN_{pc}_grid')(concat_each_pc_grid)
        return tcn_pc_grid_output

    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')
    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')

    pc_6010_conv_out = local_convolution_TCN(pc_6010, grid_input, 6010)
    pc_6014_conv_out = local_convolution_TCN(pc_6014, grid_input, 6014)
    pc_6011_conv_out = local_convolution_TCN(pc_6011, grid_input, 6011)
    pc_6280_conv_out = local_convolution_TCN(pc_6280, grid_input, 6280)
    pc_6281_conv_out = local_convolution_TCN(pc_6281, grid_input, 6281)
    pc_6284_conv_out = local_convolution_TCN(pc_6284, grid_input, 6284)

    concat_features = layers.concatenate(
        [pc_6010_conv_out, pc_6014_conv_out, pc_6011_conv_out, pc_6280_conv_out, pc_6281_conv_out, pc_6284_conv_out],
        name='concatenate_all')
    flatten_out = layers.Flatten(name='flatten_all')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    pc_and_grid_input_together_model = keras.Model(
        inputs=[grid_input, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer)

    pc_and_grid_input_together_model.compile(loss=tf.losses.MeanSquaredError(),
                                             optimizer=tf.optimizers.Adam(0.0001),
                                             metrics=[tf.metrics.MeanAbsoluteError()])
    return pc_and_grid_input_together_model


def grid_added_at_each_TCN_together():
    def get_tcn_layer(dilation_rate, pc, layer_num, input_to_layer):
        return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name=f'TCN_{layer_num}_{pc}_grid')(input_to_layer)

    def local_convolution_TCN(pc_ts, grid_ts, pc):
        cnn_layer = 6
        dilation_rate = 2
        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
        concat_each_pc_grid = layers.concatenate([pc_ts, grid_ts], name=f'concat_{pc}_grid')

        tcn_layer1 = get_tcn_layer([dilation_rates[0]], pc, 1, concat_each_pc_grid)
        concat_grid_with_layer1 = layers.concatenate([tcn_layer1, grid_ts])
        tcn_layer2 = get_tcn_layer([dilation_rates[1]], pc, 2, concat_grid_with_layer1)
        concat_grid_with_layer2 = layers.concatenate([tcn_layer2, grid_ts])
        tcn_layer3 = get_tcn_layer([dilation_rates[2]], pc, 3, concat_grid_with_layer2)
        concat_grid_with_layer3 = layers.concatenate([tcn_layer3, grid_ts])
        tcn_layer4 = get_tcn_layer([dilation_rates[3]], pc, 4, concat_grid_with_layer3)

        return tcn_layer4

    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')
    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')

    pc_6010_conv_out = local_convolution_TCN(pc_6010, grid_input, 6010)
    pc_6014_conv_out = local_convolution_TCN(pc_6014, grid_input, 6014)
    pc_6011_conv_out = local_convolution_TCN(pc_6011, grid_input, 6011)
    pc_6280_conv_out = local_convolution_TCN(pc_6280, grid_input, 6280)
    pc_6281_conv_out = local_convolution_TCN(pc_6281, grid_input, 6281)
    pc_6284_conv_out = local_convolution_TCN(pc_6284, grid_input, 6284)

    concat_features = layers.concatenate(
        [pc_6010_conv_out, pc_6014_conv_out, pc_6011_conv_out, pc_6280_conv_out, pc_6281_conv_out, pc_6284_conv_out],
        name='concatenate_all')
    flatten_out = layers.Flatten(name='flatten_all')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    grid_added_at_each_TCN_together_model = keras.Model(
        inputs=[grid_input, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer)

    grid_added_at_each_TCN_together_model.compile(loss=tf.losses.MeanSquaredError(),
                                                  optimizer=tf.optimizers.Adam(0.0001),
                                                  metrics=[tf.metrics.MeanAbsoluteError()])
    return grid_added_at_each_TCN_together_model


def grid_conv_added_at_each_TCN_together():
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
                       use_skip_connections=False, dropout_rate=0.05,
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


def grid_only_network():
    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
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
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_grid)
    grid_only_network_model = keras.Model(grid_input, full_connected_layer)
    grid_only_network_model.compile(loss=tf.losses.MeanSquaredError(),
                                    optimizer=tf.optimizers.Adam(0.0001),
                                    metrics=[tf.metrics.MeanAbsoluteError()])
    return grid_only_network_model

def local_and_global_conv_approach_with_TCN():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')

    # postcode convolutions
    concatenation_pc = layers.concatenate([pc_6010, pc_6014, pc_6011, pc_6280, pc_6281, pc_6284],
                                          name='postcode_concat')
    pc_normalization = layers.LayerNormalization()(concatenation_pc)
    cnn_layer = 6
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    padding = 'causal'
    use_skip_connections = False
    return_sequences = True
    dropout_rate = 0.05
    kernel_initializer = 'he_normal'
    activation = 'relu'
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                          padding=padding,
                          use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                          return_sequences=return_sequences,
                          activation=activation, kernel_initializer=kernel_initializer,
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(pc_normalization)

    # concatenation
    concat_grid_conv_pc_conv = layers.concatenate([input_grid, tcn_pc_grid], name='grid_pcTCN_concat')
    concat_grid_conv_pc_conv_normal = layers.LayerNormalization()(concat_grid_conv_pc_conv)

    # Convolution
    cnn_layer_full = 6
    dilation_rate_full = 2
    dilation_rates_full = [dilation_rate_full ** i for i in range(cnn_layer_full)]
    tcn_full = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer_full, dilations=dilation_rates_full,
                       padding=padding,
                       use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                       return_sequences=return_sequences,
                       activation=activation, kernel_initializer=kernel_initializer,
                       use_batch_norm=use_batch_norm,
                       use_layer_norm=use_layer_norm,
                       use_weight_norm=use_weight_norm, name='full_TCN')(concat_grid_conv_pc_conv_normal)

    # Fully Connected Layer
    flatten_layer = layers.Flatten(name='flatten_all')(tcn_full)
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_layer)

    local_and_global_conv_approach_with_TCN_model = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    local_and_global_conv_approach_with_TCN_model.compile(loss=tf.losses.MeanSquaredError(),
                                    optimizer=tf.optimizers.Adam(0.0001),
                                    metrics=[tf.metrics.MeanAbsoluteError()])
    return local_and_global_conv_approach_with_TCN_model


def frozen_branch_approach_TCN():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')

    # postcode convolutions
    concatenation_pc = layers.concatenate([pc_6010, pc_6014, pc_6011, pc_6280, pc_6281, pc_6284],
                                          name='postcode_concat')
    pc_normalization = layers.LayerNormalization()(concatenation_pc)
    cnn_layer = 6
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    padding = 'causal'
    use_skip_connections = False
    return_sequences = True
    dropout_rate = 0.05
    kernel_initializer = 'he_normal'
    activation = 'relu'
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = True
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                          padding=padding,
                          use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                          return_sequences=return_sequences,
                          activation=activation, kernel_initializer=kernel_initializer,
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(pc_normalization)
    flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    full_connected_layer_pc = layers.Dense(14, activation='linear', name="prediction_layer_pc")(flatten_pc)

    # LOAD PRETRAINED GRID MODEL
    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')
    grid_network = tf.keras.models.load_model('combined_nn_results/refined_models/multiple_runs/saved_models/0')
    grid_network.trainable = False
    grid_model = grid_network(input_grid, training=False)

    concatenation = layers.concatenate([grid_model, full_connected_layer_pc])
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(concatenation)

    frozen_branch_approach_TCN_model = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    frozen_branch_approach_TCN_model.compile(loss=tf.losses.MeanSquaredError(),
                                optimizer=tf.optimizers.Adam(0.0001),
                                metrics=[tf.metrics.MeanAbsoluteError()])
    return frozen_branch_approach_TCN_model