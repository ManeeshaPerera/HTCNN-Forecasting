import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn
from constants import SWIS_POSTCODES


# Changing the best approach of Approach A so far
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


def approachA_increase_number_of_filters():
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
    tcn_pc_grid = tcn.TCN(nb_filters=64, kernel_size=2, nb_stacks=tcn_stacks, dilations=dilation_rates,
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
    approachA_increase_filter_size_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    approachA_increase_filter_size_model.compile(loss=tf.losses.MeanSquaredError(),
                                                 optimizer=tf.optimizers.Adam(0.0001),
                                                 metrics=[tf.metrics.MeanAbsoluteError()])
    return approachA_increase_filter_size_model


def approachA_increase_layers():
    input_layers_pc = []
    for ts in SWIS_POSTCODES:
        input_layer = keras.Input(shape=(18 * 1, 14), name=f'input_postcode_{ts}')
        input_layers_pc.append(input_layer)

    # postcode convolutions
    concatenation_pc = layers.concatenate(input_layers_pc,
                                          name='postcode_concat')
    # pc_normalization = layers.LayerNormalization()(concatenation_pc)
    cnn_layer = 14
    tcn_stacks = 10
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
    approachA_increase_layers_model = keras.Model(
        inputs=input_layers_pc, outputs=prediction_layer)

    approachA_increase_layers_model.compile(loss=tf.losses.MeanSquaredError(),
                                            optimizer=tf.optimizers.Adam(0.0001),
                                            metrics=[tf.metrics.MeanAbsoluteError()])
    return approachA_increase_layers_model


def grid_only_network_reshape():
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    cnn_layer = 6
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=4, dilations=dilation_rates,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    # flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    full_connected_layer = layers.Dense(1, activation='linear', name="prediction_layer_grid")(tcn_grid)
    grid_only_network_model = keras.Model(grid_input, full_connected_layer)
    return grid_only_network_model


def SWIS_APPROACH_A_reshape_appraoch():
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
    # flatten_pc = layers.Flatten(name='flatten_pc')(tcn_pc_grid)
    full_connected_layer_pc = layers.Dense(1, activation='linear', name="prediction_layer_pc")(tcn_pc_grid)

    grid_model = grid_only_network_reshape()

    concatenation = layers.concatenate([grid_model.output, full_connected_layer_pc])
    prediction_layer = layers.Dense(1, activation='linear', name="prediction_layer")(concatenation)
    reshape_pred = layers.Reshape((18,))(prediction_layer)

    input_layers_pc.append(grid_model.input)
    SWIS_APPROACH_A_reshape_appraoch_model = keras.Model(
        inputs=input_layers_pc, outputs=reshape_pred)

    SWIS_APPROACH_A_reshape_appraoch_model.compile(loss=tf.losses.MeanSquaredError(),
                                                   optimizer=tf.optimizers.Adam(0.0001),
                                                   metrics=[tf.metrics.MeanAbsoluteError()])
    return SWIS_APPROACH_A_reshape_appraoch_model


def simple_grid_network():
    grid_input = keras.Input(shape=(18 * 1, 7), name='input_grid')
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(10)]
    tcn_grid = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=1, dilations=dilation_rates,
                       padding='causal',
                       use_skip_connections=False, dropout_rate=0.05,
                       return_sequences=True,
                       activation='relu', kernel_initializer='he_normal',
                       use_batch_norm=False,
                       use_layer_norm=False,
                       use_weight_norm=True, name='TCN_grid')(grid_input)
    # flatten_grid = layers.Flatten(name='flatten_grid')(tcn_grid)
    full_connected_layer = layers.Dense(1, activation='linear', name="prediction_layer_grid")(tcn_grid)
    simple_grid_network_model = keras.Model(grid_input, full_connected_layer)
    return simple_grid_network_model


def pc_together_2D_conv_approach_with_grid():
    pc_input = keras.Input(shape=(18 * 1, 1 * 101, 14), name='input_pc')

    conv_2d_layer1 = layers.Conv2D(32, kernel_size=(4, 4))(pc_input)
    conv_2d_layer2 = layers.Conv2D(32, kernel_size=(4, 4))(conv_2d_layer1)
    flatten_out = layers.Flatten(name='flatten_all')(conv_2d_layer2)
    full_connected_layer_pc = layers.Dense(18, activation='linear', name="prediction_layer_pc")(flatten_out)
    reshape_pc = layers.Reshape((18, 1))(full_connected_layer_pc)

    grid_model = simple_grid_network()

    concatenation = layers.concatenate([grid_model.output, reshape_pc])
    prediction_layer = layers.Dense(1, activation='linear', name="prediction_layer")(concatenation)
    reshape_pred = layers.Reshape((18,))(prediction_layer)

    pc_together_2D_conv_approach_model = keras.Model(
        inputs=[grid_model.input, pc_input], outputs=reshape_pred)

    pc_together_2D_conv_approach_model.compile(loss=tf.losses.MeanSquaredError(),
                                               optimizer=tf.optimizers.Adam(0.0001),
                                               metrics=[tf.metrics.MeanAbsoluteError()])
    return pc_together_2D_conv_approach_model
