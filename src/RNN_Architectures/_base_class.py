import tensorflow as tf


class RNN:
    def __init__(self, epochs, window_generator, lr):
        self.epochs = epochs
        self.window_generator = window_generator
        self.lr = lr

    def fit(self, model):
        history = model.fit(self.window_generator.train, epochs=self.epochs)
        print(model.summary())
        return history

    def evaluate_performance(self, model):
        performance = {'val': model.evaluate(self.window_generator.val),
                       'test': model.evaluate(self.window_generator.test, verbose=0)}
        return performance

    def forecast(self, model):
        forecast = []
        actual = []
        for sample_input, sample_output in self.window_generator.test:
            fc = model.predict(sample_input)
            forecast.append(fc)
            actual.append(sample_output.numpy()[:, :, 0])
        return forecast, actual

    def evaluate_model(self, model):
        print("evaluation")
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
        model.fit(self.window_generator.train, validation_data=self.window_generator.val, epochs=self.epochs, verbose=0,
                  callbacks=[callback])
        score = model.evaluate(self.window_generator.val, verbose=0)
        # Return the loss (minimize the loss)
        return -score[0]
