import tensorflow as tf


class RNN:
    def __init__(self, epochs, window_generator):
        self.epochs = epochs
        self.model = None
        self.window_generator = window_generator

    def compile_and_fit(self, patience=50):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(self.window_generator.train, epochs=self.epochs,
                                 validation_data=self.window_generator.val,
                                 callbacks=[early_stopping])
        print(self.model.summary())
        return history

    def evaluate(self):
        performance = {'val': self.model.evaluate(self.window_generator.val),
                       'test': self.model.evaluate(self.window_generator.test, verbose=0)}
        return performance

    def forecast(self, normalised_values, horizon):
        forecast = []
        for time in range(len(normalised_values) - self.window_generator.input_width, horizon):
            fc = self.model.predict(normalised_values[time:time + self.window_generator.input_width])
            forecast.append(fc)
        return forecast
