import tensorflow as tf


class RNN:
    def __init__(self, epochs, window_generator):
        self.epochs = epochs
        self.model = None
        self.window_generator = window_generator

    def compile_and_fit(self):
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(self.window_generator.train, epochs=self.epochs)
        print(self.model.summary())
        return history

    def evaluate(self):
        performance = {'val': self.model.evaluate(self.window_generator.val),
                       'test': self.model.evaluate(self.window_generator.test, verbose=0)}
        return performance

    def forecast(self):
        forecast = []
        actual = []
        for sample_input, sample_output in self.window_generator.test:
            fc = self.model.predict(sample_input)
            forecast.append(fc)
            actual.append(sample_output.numpy()[:, :, 0])
        return forecast, actual
