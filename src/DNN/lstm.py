from src.DNN._base_class import DNNModel
import tensorflow as tf
from tensorflow.keras import regularizers


class LSTMModel(DNNModel):
    def __init__(self, horizon, num_lags, data, epochs, lr, layers, cell_dim, exog=False):
        super().__init__(horizon, num_lags, data, exog)
        self.epochs = epochs
        self.lr = lr
        self.layers = layers
        self.cell_dim = cell_dim

    def compile_and_fit_lstm(self, train_X, train_Y, val_X, val_Y):
        tf.keras.backend.clear_session()
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # clearing the tensorflow session before creating a new model
            model = tf.keras.Sequential()
            if self.layers > 1:
                model.add(tf.keras.layers.LSTM(self.cell_dim, return_sequences=True,
                                               input_shape=(train_X.shape[1], train_X.shape[2]),
                                               kernel_regularizer=regularizers.l2(0.001)))
                # we are already adding the first layer and last layer so the -2
                for layer in range(self.layers - 2):
                    model.add(tf.keras.layers.LSTM(self.cell_dim, return_sequences=True,
                                                   kernel_regularizer=regularizers.l2(0.001)))
                model.add(tf.keras.layers.LSTM(self.cell_dim, return_sequences=False,
                                               kernel_regularizer=regularizers.l2(0.001)))
            else:
                model.add(tf.keras.layers.LSTM(self.cell_dim, return_sequences=False,
                                               input_shape=(train_X.shape[1], train_X.shape[2]),
                                               kernel_regularizer=regularizers.l2(0.001)))

            # convert the output to our desired horizon
            model.add(tf.keras.layers.Dense(self.horizon))
            print(model.summary())

            model.compile(loss=tf.losses.MeanSquaredError(),
                          optimizer=tf.optimizers.Adam(self.lr),
                          metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(train_X, train_Y, epochs=self.epochs, validation_data=(val_X, val_Y))
        return history, model
