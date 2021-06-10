SEED = 1234
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def local_convolution_stage(pc):
    input_postcode = tf.keras.Input(shape=(14 * 1, 14), name=f'input_postcode_{pc}')
    conv1 = tf.keras.layers.Conv1D(kernel_size=2, padding='same', filters=32, name=f'conv1_postcode_{pc}', activation='relu')(
        input_postcode)
    max_pool = tf.keras.layers.MaxPooling1D(padding='same', strides=1)(conv1)
    model_pc = tf.keras.Model(input_postcode, max_pool)
    return model_pc



def local_and_full_convolution_approach_alternative1():
    # removing the Grid level Branch
    pc_6010 = local_convolution_stage(6010)
    pc_6014 = local_convolution_stage(6014)
    pc_6011 = local_convolution_stage(6011)
    pc_6280 = local_convolution_stage(6280)
    pc_6281 = local_convolution_stage(6281)
    pc_6284 = local_convolution_stage(6284)

    concatenation = tf.keras.layers.concatenate(
        [pc_6010.output, pc_6014.output, pc_6011.output, pc_6280.output, pc_6281.output,
         pc_6284.output])
    conv_full_stage = tf.keras.layers.Conv1D(kernel_size=4, filters=32, name='conv1_full_stage')(concatenation)
    conv2_full_stage = tf.keras.layers.Conv1D(kernel_size=4, filters=32, name='conv2_full_stage')(conv_full_stage)
    max_pool_full_stage = tf.keras.layers.MaxPooling1D(padding='same', name='max_pool_full_stage')(conv2_full_stage)
    flatten_out = tf.keras.layers.Flatten(name='flatten_full_stage')(max_pool_full_stage)
    full_connected_layer = tf.keras.layers.Dense(14, activation='linear', name="prediction_layer")(flatten_out)

    local_and_full_conv_model = tf.keras.Model(
        inputs=[pc_6010.input, pc_6014.input, pc_6011.input, pc_6280.input, pc_6281.input,
                pc_6284.input], outputs=full_connected_layer)

    local_and_full_conv_model.compile(loss=tf.losses.MeanSquaredError(),
                                      optimizer=tf.optimizers.Adam(0.0001),
                                      metrics=[tf.metrics.MeanAbsoluteError()])
    return local_and_full_conv_model


