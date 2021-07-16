import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn
from constants import SWIS_POSTCODES


def SWIS_APPROACH_A_more_layer():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    pc_normalization = layers.LayerNormalization()(concatenation_pc)
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
                          use_weight_norm=use_weight_norm, name='pc_TCN')(pc_normalization)
    flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    full_connected_layer_pc = layers.Dense(18, activation='linear', name="prediction_layer_pc")(flatten_pc)

    grid_model = grid_only_network_SWIS()

    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_model.input)
    SWIS_APPROACH_A_more_layer_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    SWIS_APPROACH_A_more_layer_model.compile(loss=tf.losses.MeanSquaredError(),
                                             optimizer=tf.optimizers.Adam(0.0001),
                                             metrics=[tf.metrics.MeanAbsoluteError()])
    return SWIS_APPROACH_A_more_layer_model


def SWIS_APPROACH_A_more_layer_without_norm():
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

    grid_model = grid_only_network_SWIS()

    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_model.input)
    SWIS_APPROACH_A_more_layer_without_norm_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    SWIS_APPROACH_A_more_layer_without_norm_model.compile(loss=tf.losses.MeanSquaredError(),
                                                          optimizer=tf.optimizers.Adam(0.0001),
                                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return SWIS_APPROACH_A_more_layer_without_norm_model


def gride_level_simple_CNN():
    input_data = keras.Input(shape=(18 * 1, 14), name=f'input_grid')
    cnn_layer1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer1')(input_data)
    cnn_layer2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer2')(cnn_layer1)
    max_pool_stage = layers.MaxPooling1D(padding='same')(cnn_layer2)
    cnn_layer3 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer3')(max_pool_stage)
    cnn_layer4 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer4')(cnn_layer3)
    max_pool_stage_2 = layers.MaxPooling1D(padding='same')(cnn_layer4)

    flatten_out = layers.Flatten(name='flatten_layer')(max_pool_stage_2)
    #change for swis - we need to forecast 18 points
    fully_connect_layer = layers.Dense(18, activation='linear')(flatten_out)
    conventional_CNN = keras.Model(
        inputs=input_data, outputs=fully_connect_layer)

    conventional_CNN.compile(loss=tf.losses.MeanSquaredError(),
                             optimizer=tf.optimizers.Adam(0.0001),
                             metrics=[tf.metrics.MeanAbsoluteError()])
    return conventional_CNN



def SWIS_APPROACH_A_more_layer_with_simple_CNN():
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

    grid_model = gride_level_simple_CNN()

    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_model.input)
    SWIS_APPROACH_A_more_layer_without_norm_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    SWIS_APPROACH_A_more_layer_without_norm_model.compile(loss=tf.losses.MeanSquaredError(),
                                                          optimizer=tf.optimizers.Adam(0.0001),
                                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return SWIS_APPROACH_A_more_layer_without_norm_model


# WITH SKIP CONNECTIONS
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


def SWIS_APPROACH_A_SKIP():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    pc_normalization = layers.LayerNormalization()(concatenation_pc)
    cnn_layer = 6
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
    tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                          padding=padding,
                          use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                          return_sequences=return_sequences,
                          activation=activation, kernel_initializer=kernel_initializer,
                          use_batch_norm=use_batch_norm,
                          use_layer_norm=use_layer_norm,
                          use_weight_norm=use_weight_norm, name='pc_TCN')(pc_normalization)
    flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    full_connected_layer_pc = layers.Dense(18, activation='linear', name="prediction_layer_pc")(flatten_pc)

    grid_model = grid_only_network_SWIS()

    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_model.input)
    frozen_branch_approach_TCN_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    frozen_branch_approach_TCN_model.compile(loss=tf.losses.MeanSquaredError(),
                                             optimizer=tf.optimizers.Adam(0.0001),
                                             metrics=[tf.metrics.MeanAbsoluteError()])
    return frozen_branch_approach_TCN_model


# WITHOUT SKIP CONNECTIONS
def grid_only_network_SWIS():
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
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
    full_connected_layer = layers.Dense(18, activation='linear', name="prediction_layer_grid")(flatten_grid)
    grid_only_network_model = keras.Model(grid_input, full_connected_layer)
    return grid_only_network_model


def SWIS_APPROACH_A():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
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
    full_connected_layer_pc = layers.Dense(18, activation='linear', name="prediction_layer_pc")(flatten_pc)

    grid_model = grid_only_network_SWIS()

    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(18, activation='linear', name="prediction_layer")(concatenation)

    input_layers_pc.append(grid_model.input)
    frozen_branch_approach_TCN_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    frozen_branch_approach_TCN_model.compile(loss=tf.losses.MeanSquaredError(),
                                             optimizer=tf.optimizers.Adam(0.0001),
                                             metrics=[tf.metrics.MeanAbsoluteError()])
    return frozen_branch_approach_TCN_model


# postcode level branch
def postcode_level_branch_approachA():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    # postcode convolutions
    concatenation_pc = layers.concatenate([pc_6010, pc_6014, pc_6011, pc_6280, pc_6281, pc_6284],
                                          name='postcode_concat')
    pc_normalization = layers.LayerNormalization(name='postcode_concat_normalize')(concatenation_pc)
    # cnn_layer = 6
    # dilation_rate = 2
    # dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    # padding = 'causal'
    # use_skip_connections = False
    # return_sequences = True
    # dropout_rate = 0.05
    # kernel_initializer = 'he_normal'
    # activation = 'relu'
    # use_batch_norm = False
    # use_layer_norm = False
    # use_weight_norm = True
    # tcn_pc_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
    #                       padding=padding,
    #                       use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
    #                       return_sequences=return_sequences,
    #                       activation=activation, kernel_initializer=kernel_initializer,
    #                       use_batch_norm=use_batch_norm,
    #                       use_layer_norm=use_layer_norm,
    #                       use_weight_norm=use_weight_norm, name='pc_TCN')(pc_normalization)

    cnn_layer1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'pc_all_cnn_layer1')(pc_normalization)
    cnn_layer2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'pc_all_cnn_layer2')(cnn_layer1)
    max_pool_stage = layers.MaxPooling1D(padding='same', name='pc_all_max_pool_1')(cnn_layer2)
    cnn_layer3 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'pc_all_cnn_layer3')(max_pool_stage)
    cnn_layer4 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'pc_all_cnn_layer4')(cnn_layer3)
    max_pool_stage_2 = layers.MaxPooling1D(padding='same', name='pc_all_max_pool_2')(cnn_layer4)
    flatten_pc = layers.Flatten(name='flatten_pc_all')(max_pool_stage_2)

    # flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    full_connected_layer_pc = layers.Dense(14, activation='linear', name="prediction_layer_pc_all")(flatten_pc)

    postcode_level_branch_model = keras.Model(
        inputs=[pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer_pc, name='PC_ALL_MODEL_APPROACH_A')

    postcode_level_branch_model.compile(loss=tf.losses.MeanSquaredError(),
                                        optimizer=tf.optimizers.Adam(0.0001),
                                        metrics=[tf.metrics.MeanAbsoluteError()])
    return postcode_level_branch_model


def grid_level_branch_approachA():
    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
    # pass the grid input with Convolution
    # cnn_layer = 4
    # dilation_rate = 2
    # dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    # tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
    #                    padding='causal',
    #                    use_skip_connections=False, dropout_rate=0.05,
    #                    return_sequences=True,
    #                    activation='relu', kernel_initializer='he_normal',
    #                    use_batch_norm=False,
    #                    use_layer_norm=False,
    #                    use_weight_norm=True, name='TCN_grid')(grid_input)
    # flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    cnn_layer1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer1_grid')(grid_input)
    cnn_layer2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer2_grid')(cnn_layer1)
    max_pool_stage = layers.MaxPooling1D(padding='same', name='cnn_max_pool_layer1_grid')(cnn_layer2)
    cnn_layer3 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer3_grid')(max_pool_stage)
    cnn_layer4 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer4_grid')(cnn_layer3)
    max_pool_stage_2 = layers.MaxPooling1D(padding='same', name='cnn_max_pool_layer2_grid')(cnn_layer4)
    flatten_grid = layers.Flatten(name='flatten_grid')(max_pool_stage_2)

    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer_grid")(flatten_grid)
    grid_only_network_model = keras.Model(grid_input, full_connected_layer, name='GRID_MODEL_APPROACH_A')
    grid_only_network_model.compile(loss=tf.losses.MeanSquaredError(),
                                    optimizer=tf.optimizers.Adam(0.0001),
                                    metrics=[tf.metrics.MeanAbsoluteError()])

    return grid_only_network_model


def possibility_2_ApproachA():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

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
    grid_network = tf.keras.models.load_model(
        'combined_nn_results/refined_models/pre_trained_models/CNN/grid/saved_models/grid_level_branch_approachA/0')
    grid_network.trainable = False
    grid_model = grid_network(input_grid, training=False)

    concatenation = layers.concatenate([grid_model, full_connected_layer_pc])
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(concatenation)

    possibility_2_ApproachA_model = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    possibility_2_ApproachA_model.compile(loss=tf.losses.MeanSquaredError(),
                                          optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return possibility_2_ApproachA_model


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
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer_grid")(flatten_grid)
    grid_only_network_model = keras.Model(grid_input, full_connected_layer)
    return grid_only_network_model


def possibility_3_ApproachA():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    # LOAD PRETRAINED PC MODEL
    pc_model = tf.keras.models.load_model(
        'combined_nn_results/refined_models/pre_trained_models/CNN/all_pc/saved_models/postcode_level_branch_approachA/0')
    pc_model.trainable = False
    pc_output = pc_model([pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                          pc_6284], training=False)

    grid_model = grid_only_network()
    concatenation = layers.concatenate([grid_model.output, pc_output])
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(concatenation)

    possibility_3_ApproachA_model = keras.Model(
        inputs=[grid_model.input, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    possibility_3_ApproachA_model.compile(loss=tf.losses.MeanSquaredError(),
                                          optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return possibility_3_ApproachA_model


def possibility_4_ApproachA():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    # LOAD PRETRAINED PC MODEL
    pc_model = tf.keras.models.load_model(
        'combined_nn_results/refined_models/pre_trained_models/CNN/all_pc/saved_models/postcode_level_branch_approachA/0')
    pc_model.trainable = False
    pc_output = pc_model([pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                          pc_6284], training=False)

    # LOAD PRETRAINED GRID MODEL
    input_grid = keras.Input(shape=(14 * 1, 7), name='input_grid')
    grid_network = tf.keras.models.load_model(
        'combined_nn_results/refined_models/pre_trained_models/CNN/grid/saved_models/grid_level_branch_approachA/0')
    grid_network.trainable = False
    grid_model = grid_network(input_grid, training=False)

    concatenation = layers.concatenate([grid_model, pc_output])
    prediction_layer = layers.Dense(14, activation='linear', name="prediction_layer")(concatenation)

    possibility_4_ApproachA_model = keras.Model(
        inputs=[input_grid, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=prediction_layer)

    possibility_4_ApproachA_model.compile(loss=tf.losses.MeanSquaredError(),
                                          optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return possibility_4_ApproachA_model
