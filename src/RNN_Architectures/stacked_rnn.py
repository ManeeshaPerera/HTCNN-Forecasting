from src.RNN_Architectures._base_class import RNN
import tensorflow as tf


class StackedRNN(RNN):
    def __init__(self, lstm_layers, output_steps, num_features, epochs, window_generator):
        super().__init__(epochs, window_generator)
        self.lstm_layers = lstm_layers
        self.output_steps = output_steps
        self.num_features = num_features

    def create_model(self):
        model = tf.keras.Sequential()
        for layer in range(self.lstm_layers - 1):
            model.add(tf.keras.layers.LSTM(32, return_sequences=True))
        model.add(tf.keras.layers.LSTM(32, return_sequences=False))
        # Shape => [batch, out_steps*features]
        model.add(tf.keras.layers.Dense(self.output_steps * self.num_features,
                                        kernel_initializer=tf.initializers.zeros))
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([self.output_steps, self.num_features])

        self.model = model
