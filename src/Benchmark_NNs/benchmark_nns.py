import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import src.CNN_architectures.temporal_conv as tcn
from tensorflow.keras import regularizers


def lstm_model_approach(input_shape, ts):
    input_data = keras.Input(shape=input_shape, name=f'input_{ts}')
    lstm_layer1 = layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2())(input_data)
    lstm_layer2 = layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2())(lstm_layer1)
    lstm_layer3 = layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2())(lstm_layer2)
    fully_connect_layer = layers.Dense(14, activation='linear')(lstm_layer3)
    lstm_model = keras.Model(
        inputs=input_data, outputs=fully_connect_layer)

    lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                       optimizer=tf.optimizers.Adam(0.0001),
                       metrics=[tf.metrics.MeanAbsoluteError()])
    return lstm_model


def conventional_CNN_approach(input_shape, ts):
    input_data = keras.Input(shape=input_shape, name=f'input_{ts}')
    cnn_layer1 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer1')(input_data)
    cnn_layer2 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer2')(cnn_layer1)
    max_pool_stage = layers.MaxPooling1D(padding='same')(cnn_layer2)
    cnn_layer3 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer3')(max_pool_stage)
    cnn_layer4 = layers.Conv1D(kernel_size=2, padding='causal', filters=32, name=f'cnn_layer4')(cnn_layer3)
    max_pool_stage_2 = layers.MaxPooling1D(padding='same')(cnn_layer4)

    flatten_out = layers.Flatten(name='flatten_layer')(max_pool_stage_2)
    fully_connect_layer = layers.Dense(14, activation='linear')(flatten_out)
    conventional_CNN = keras.Model(
        inputs=input_data, outputs=fully_connect_layer)

    conventional_CNN.compile(loss=tf.losses.MeanSquaredError(),
                             optimizer=tf.optimizers.Adam(0.0001),
                             metrics=[tf.metrics.MeanAbsoluteError()])
    return conventional_CNN


def conventional_TCN_approach(input_shape, ts):
    input_data = keras.Input(shape=input_shape, name=f'input_{ts}')
    cnn_layer = 4
    dilation_rate = 2
    dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
    # skip connections are True in conventional TCN
    tcn_output = tcn.TCN(nb_filters=32, kernel_size=2, nb_stacks=cnn_layer, dilations=dilation_rates,
                         padding='causal',
                         use_skip_connections=True, dropout_rate=0.05,
                         return_sequences=True,
                         activation='relu', kernel_initializer='he_normal',
                         use_batch_norm=False,
                         use_layer_norm=False,
                         use_weight_norm=True, name=f'TCN_{ts}')(input_data)
    flatten_out = layers.Flatten(name='flatten_layer')(tcn_output)
    full_connected_layer = layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)
    tcn_model = keras.Model(input_data, full_connected_layer)

    tcn_model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(0.0001),
                      metrics=[tf.metrics.MeanAbsoluteError()])
    return tcn_model
