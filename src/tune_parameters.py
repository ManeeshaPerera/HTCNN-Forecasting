from bayes_opt import BayesianOptimization
from src.RNN_Architectures.stacked_rnn import StackedRNN
from src.WindowGenerator.window_generator import WindowGenerator


class TuneHyperParameters:
    def __init__(self, output_steps, num_features, seasonality, train, val, test, col_name):
        self.output_steps = output_steps
        self.num_features = num_features
        self.seasonality = seasonality
        self.train = train
        self.val = val
        self.test = test
        self.col_name = col_name

    def tune_parameters(self):
        # Bounded region of parameter space
        look_back_min = 1.25 * self.output_steps
        look_back_max = self.seasonality * 1.25

        # min_batch_min = len(self.train) * 0.1
        # min_batch_max = min_batch_min + 100
        pbounds = {'lr': (1e-3, 1e-1), 'num_layers': (1, 5), 'cell_dimension': (20, 50), 'epochs': (10, 50),
                   'look_back': (look_back_min, look_back_max)}

        optimizer = BayesianOptimization(
            f=self.fit_and_evaluate,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )

        optimizer.maximize(init_points=5, n_iter=10)
        model_params = optimizer.max['params']
        print("final hyper-parameters=", model_params)
        return model_params

    def fit_and_evaluate(self, lr, num_layers, cell_dimension, epochs, look_back):
        # Create the model using a specified hyper-parameters
        batch_size = 128
        window_data = WindowGenerator(int(look_back), self.output_steps, self.output_steps, self.train, self.val,
                                      self.test,
                                      batch_size=int(batch_size),
                                      label_columns=[self.col_name])
        lstm = StackedRNN(int(num_layers), self.output_steps, self.num_features, cell_dimension=int(cell_dimension),
                          epochs=int(epochs),
                          window_generator=window_data, lr=lr)
        model = lstm.create_model()
        loss = lstm.evaluate_model(model)
        return loss
