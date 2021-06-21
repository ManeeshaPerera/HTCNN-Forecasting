import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn


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

    postcode_level_branch_model = keras.Model(
        inputs=[pc_6010, pc_6014, pc_6011, pc_6280, pc_6281,
                pc_6284], outputs=full_connected_layer_pc)

    postcode_level_branch_model.compile(loss=tf.losses.MeanSquaredError(),
                                        optimizer=tf.optimizers.Adam(0.0001),
                                        metrics=[tf.metrics.MeanAbsoluteError()])
    return postcode_level_branch_model
