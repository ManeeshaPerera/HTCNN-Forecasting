from src.CNN_architectures.temporal_conv import create_model
from src.RNN_Architectures._base_class import RNN


class TCN(RNN):
    def __init__(self, cnn_layers, output_steps, num_features, n_filters, epochs, lr, window_generator, kernel_size,
                 dilation_rates, lookback):
        super().__init__(epochs, window_generator, lr)
        self.cnn_layers = cnn_layers
        self.output_steps = output_steps
        self.num_features = num_features
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.lookback = lookback

    def create_model(self):
        return create_model(num_feat=self.num_features, nb_filters=self.n_filters, kernel_size=self.kernel_size,
                            dilations=self.dilation_rates, nb_stacks=self.cnn_layers, max_len=self.lookback, lr=self.lr)
