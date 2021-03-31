from src.RNN_Architectures._base_class import RNN
import tensorflow as tf


# importing the RNN base class - this should be moved/ rename because it is a general base class

class DilatedCNN(RNN):
    def __init__(self, cnn_layers, output_steps, num_features, n_filters, epochs, lr, window_generator, kernel_size,
                 dilation_rates):
        super().__init__(epochs, window_generator, lr)
        self.cnn_layers = cnn_layers
        self.output_steps = output_steps
        self.num_features = num_features
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates

    def create_model(self):
        tf.keras.backend.clear_session()
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # clearing the tensorflow session before creating a new model
            model = tf.keras.Sequential()
            # layer1 = tf.keras.layers.Lambda(lambda x: x[:, -(self.kernel_size-1):, :])
            layer1 = tf.keras.layers.Conv1D(filters=self.n_filters,
                                            kernel_size=self.kernel_size,
                                            padding='causal',
                                            input_shape=(14 * 7, 1))

            model.add(layer1)
            for layer in range(self.cnn_layers):
                cnn_layer = tf.keras.layers.Conv1D(filters=self.n_filters,
                                                   kernel_size=self.kernel_size,
                                                   padding='causal',
                                                   dilation_rate=self.dilation_rates[layer]
                                                   )
                model.add(cnn_layer)

            model.add(tf.keras.layers.Flatten())
            # Shape => [batch, out_steps*features]
            model.add(tf.keras.layers.Dense(self.output_steps * self.num_features,
                                            kernel_initializer=tf.initializers.zeros()))
            # print(model.summary())
            # Shape => [batch, out_steps, features]
            model.add(tf.keras.layers.Reshape([self.output_steps, self.num_features]))


            model.compile(loss=tf.losses.MeanSquaredError(),
                          optimizer=tf.optimizers.Adam(self.lr),
                          metrics=[tf.metrics.MeanAbsoluteError()])
            return model
