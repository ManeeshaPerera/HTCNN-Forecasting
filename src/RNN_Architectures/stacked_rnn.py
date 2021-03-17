from src.RNN_Architectures._base_class import RNN
import tensorflow as tf


class StackedRNN(RNN):
    def __init__(self, lstm_layers, output_steps, num_features, cell_dimension, epochs, lr, window_generator):
        super().__init__(epochs, window_generator, lr)
        self.lstm_layers = lstm_layers
        self.output_steps = output_steps
        self.num_features = num_features
        self.cell_dimension = cell_dimension

    def create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # clearing the tensorflow session before creating a new model
            tf.keras.backend.clear_session()
            model = tf.keras.Sequential()
            for layer in range(self.lstm_layers - 1):
                model.add(tf.keras.layers.LSTM(self.cell_dimension, return_sequences=True))
            model.add(tf.keras.layers.LSTM(self.cell_dimension, return_sequences=False))

            # Shape => [batch, out_steps*features]
            model.add(tf.keras.layers.Dense(self.output_steps * self.num_features,
                                            kernel_initializer=tf.initializers.zeros))
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.output_steps, self.num_features])

            model.compile(loss=tf.losses.MeanSquaredError(),
                          optimizer=tf.optimizers.Adam(self.lr),
                          metrics=[tf.metrics.MeanAbsoluteError()])
            return model
