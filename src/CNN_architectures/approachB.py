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
    cnn_layer2_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer2_grid')(
        cnn_layer1_grid)
    max_pool_stage = layers.MaxPooling1D(padding='same', strides=1)(cnn_layer2_grid)
    cnn_layer3_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer3_grid')(
        max_pool_stage)
    cnn_layer4_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer4_grid')(
        cnn_layer3_grid)
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
    grid_only_network_model = keras.Model(grid_input, full_connected_layer, name='GRIDMODEL')
    grid_only_network_model.compile(loss=tf.losses.MeanSquaredError(),
                                    optimizer=tf.optimizers.Adam(0.0001),
                                    metrics=[tf.metrics.MeanAbsoluteError()])

    return grid_only_network_model


def postcode_level_branch(pc):
    def get_tcn_layer(dilation_rate, pc, layer_num, input_to_layer):
        # return tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=dilation_rate,
        #                padding='causal',
        #                use_skip_connections=False, dropout_rate=0.05,
        #                return_sequences=True,
        #                activation='relu', kernel_initializer='he_normal',
        #                use_batch_norm=False,
        #                use_layer_norm=False,
        #                use_weight_norm=True, name=f'TCN_{layer_num}_{pc}_grid')(input_to_layer)

        cnn_layer1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_{layer_num}_{pc}_grid')(
            input_to_layer)
        cnn_layer2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_{layer_num}_1_{pc}_grid')(
            cnn_layer1)
        max_pool_stage = layers.MaxPooling1D(padding='same', strides=1, name=f'max_{layer_num}_{pc}_grid')(cnn_layer2)
        return max_pool_stage

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

    # grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
    # # pass the grid input with Convolution
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

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')
    # pass the grid input with Convolution
    cnn_layer1_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer1_grid_approachPC')(
        grid_input)
    cnn_layer2_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer2_grid_approachPC')(
        cnn_layer1_grid)
    max_pool_stage = layers.MaxPooling1D(padding='same', strides=1, name='max_pool_1_grid_approachPC')(cnn_layer2_grid)
    cnn_layer3_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer3_grid_approachPC')(
        max_pool_stage)
    cnn_layer4_grid = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer4_grid_approachPC')(
        cnn_layer3_grid)
    max_pool_stage_2 = layers.MaxPooling1D(padding='same', strides=1, name='max_pool_2_grid_approachPC')(
        cnn_layer4_grid)

    # pc_tcn_out = local_convolution_TCN(pc_data, tcn_grid, pc)
    pc_tcn_out = local_convolution_TCN(pc_data, max_pool_stage_2, pc)
    flatten_out = layers.Flatten(name='flatten_all_pc_grid')(pc_tcn_out)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer_pc_grid")(flatten_out)

    postcode_level_branch_approach = keras.Model(
        inputs=[grid_input, pc_data], outputs=full_connected_layer, name=f'{pc}_MODEL')

    postcode_level_branch_approach.compile(loss=tf.losses.MeanSquaredError(),
                                           optimizer=tf.optimizers.Adam(0.0001),
                                           metrics=[tf.metrics.MeanAbsoluteError()])
    return postcode_level_branch_approach


def load_postcode_models(pc, input_grid, pc_data):
    # LOAD PRETRAINED PC MODEL
    # layer_name = f'TCN_{4}_{pc}_grid'
    layer_name = f'max_{4}_{pc}_grid'
    pc_model = tf.keras.models.load_model(
        f'combined_nn_results/refined_models/pre_trained_models/CNN/{pc}/saved_models/postcode_level_branch/0')
    pc_model.trainable = False

    conv_output = pc_model.get_layer(layer_name).output
    pc_model([input_grid, pc_data], training=False)

    model = keras.Model(pc_model.input, outputs=conv_output)
    return model

    # intermediate_layer_model = keras.Model(inputs=[input_grid, pc_data],
    #                                        outputs=pc_model.get_layer(layer_name).output)
    # # Extract the last CNN layer
    # return intermediate_layer_model


def possibility_3_approachB():
    pc_6010 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6010')
    pc_6014 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6014')
    pc_6011 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6011')
    pc_6280 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6280')
    pc_6281 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6281')
    pc_6284 = keras.Input(shape=(14 * 1, 14), name='input_postcode_6284')

    grid_input = keras.Input(shape=(14 * 1, 7), name='input_grid')

    # LOAD PRETRAINED PC MODEL
    pc_6010_model = load_postcode_models(6010, grid_input, pc_6010)
    pc_6014_model = load_postcode_models(6014, grid_input, pc_6014)
    pc_6011_model = load_postcode_models(6011, grid_input, pc_6011)
    pc_6280_model = load_postcode_models(6280, grid_input, pc_6280)
    pc_6281_model = load_postcode_models(6281, grid_input, pc_6281)
    pc_6284_model = load_postcode_models(6284, grid_input, pc_6284)

    concat_features = layers.concatenate(
        [pc_6010_model.output, pc_6014_model.output, pc_6011_model.output, pc_6280_model.output, pc_6281_model.output,
         pc_6284_model.output],
        name='concatenate_all_conv_output')
    flatten_out = layers.Flatten(name='flatten_all_conv_output')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer_conv_output_pc")(flatten_out)

    possibility_3_approachB_model = keras.Model(
        inputs=[pc_6010_model.input, pc_6014_model.input, pc_6011_model.input, pc_6280_model.input, pc_6281_model.input,
                pc_6284_model.input], outputs=full_connected_layer)

    possibility_3_approachB_model.compile(loss=tf.losses.MeanSquaredError(),
                                          optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return possibility_3_approachB_model


def possibility_2_approachB():
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
    # LOAD PRETRAINED GRID MODEL
    grid_network = tf.keras.models.load_model(
        'combined_nn_results/refined_models/pre_trained_models/CNN/grid/saved_models/grid_level_branch/0')
    grid_network.trainable = False
    conv_output = grid_network.get_layer('cnn_max_pool_layer2_grid').output
    grid_network(grid_input, training=False)

    grid_cnn_output = keras.Model(grid_input.input, outputs=conv_output)

    pc_6010_tcn_out = local_convolution_TCN(pc_6010, grid_cnn_output.output, 6010)
    pc_6014_tcn_out = local_convolution_TCN(pc_6014, grid_cnn_output.output, 6014)
    pc_6011_tcn_out = local_convolution_TCN(pc_6011, grid_cnn_output.output, 6011)
    pc_6280_tcn_out = local_convolution_TCN(pc_6280, grid_cnn_output.output, 6280)
    pc_6281_tcn_out = local_convolution_TCN(pc_6281, grid_cnn_output.output, 6281)
    pc_6284_tcn_out = local_convolution_TCN(pc_6284, grid_cnn_output.output, 6284)

    concat_features = layers.concatenate(
        [pc_6010_tcn_out, pc_6014_tcn_out, pc_6011_tcn_out, pc_6280_tcn_out, pc_6281_tcn_out, pc_6284_tcn_out],
        name='concatenate_all')
    flatten_out = layers.Flatten(name='flatten_all')(concat_features)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    possibility_2_approachB_model = keras.Model(
        inputs=[grid_input, pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer)

    possibility_2_approachB_model.compile(loss=tf.losses.MeanSquaredError(),
                                          optimizer=tf.optimizers.Adam(0.0001),
                                          metrics=[tf.metrics.MeanAbsoluteError()])
    return possibility_2_approachB
