import pandas as pd
from sklearn.preprocessing import StandardScaler
import constants


class DNNModel():
    def __init__(self, horizon, num_lags, data, exog=False):
        self.horizon = horizon
        self.num_lags = num_lags
        self.data = data
        self.exog = exog
        self.scaler = None
        self.train = None
        self.val = None
        self.test = None

    def get_lags(self, dataframe):
        dfs = []
        # we are generating the past time lags to consider
        for i in range(1, self.num_lags + 1):
            shifted_df = dataframe.shift(i)
            shifted_df.columns = [col + "_" + str(-i) for col in dataframe.columns]
            dfs.append(shifted_df.iloc[:, 0:])

        # to the end we want to add our original data frame
        dfs.append(dataframe)
        new_df = pd.concat(dfs, axis=1).dropna()
        return new_df

    def get_lags_exogenous(self, dataframe):
        dfs = []
        for i in range(1, self.num_lags + 1):
            df1 = dataframe[['power']].shift(i)
            df1.columns = [col + "_" + str(-i) for col in ['power']]
            dfs.append(df1.iloc[:, 0:])

        # we are now adding the dataframe with weather variables
        dfs.append(dataframe.iloc[:, 1:])
        for i in range(1, self.horizon):
            df2 = dataframe.iloc[:, 1:].shift(-i)
            df2.columns = [col + "_" + str(i) for col in dataframe.iloc[:, 1:].columns]
            dfs.append(df2.iloc[:, 0:])
        # finally we add the power values
        dfs.append(dataframe[['power']])
        new_df = pd.concat(dfs, axis=1).dropna()
        return new_df

    def get_output(self, dataframe):
        dfs = [dataframe]
        for i in range(1, self.horizon):
            df1 = dataframe[['power']].shift(-i)
            df1.columns = [col + "_" + str(i) for col in ['power']]
            dfs.append(df1.iloc[:, 0:])
        return pd.concat(dfs, axis=1).dropna()

    def create_transform_array(self, scaler, dataframe):
        train_array = scaler.transform(dataframe.values)
        return pd.DataFrame(train_array, columns=dataframe.columns)

    def process_data_dnn(self):
        train = self.data[0:-14 * (constants.VAL_DAYS + constants.TEST_DAYS)]
        val = self.data[:-14 * constants.TEST_DAYS]
        test = self.data.copy()

        # scaling the data
        scaler = StandardScaler()
        scaler.fit(train.values)

        scaler_power = StandardScaler()
        scaler_power.fit(train[['power']].values)
        self.scaler = scaler_power

        train_df = self.create_transform_array(scaler, train)
        val_df = self.create_transform_array(scaler, val)
        test_df = self.create_transform_array(scaler, test)

        if self.exog:
            train_lag = self.get_lags_exogenous(train_df)
            val_lag = self.get_lags_exogenous(val_df)
            test_lag = self.get_lags_exogenous(test_df)

        else:
            train_lag = self.get_lags(train_df)
            val_lag = self.get_lags(val_df)
            test_lag = self.get_lags(test_df)

        self.train = self.get_output(train_lag)
        self.val = self.get_output(val_lag)[self.train.index[-1] - self.train.index[0] + 1:]
        self.test = self.get_output(test_lag)[self.val.index[-1] - self.train.index[0] + 1:]

    def get_train_val_test(self):
        self.process_data_dnn()
        train_X, train_Y = self.train.iloc[:, 0:-self.horizon], self.train.iloc[:, -self.horizon:]
        val_X, val_Y = self.val.iloc[:, 0:-self.horizon], self.val.iloc[:, -self.horizon:]
        test_X, test_Y = self.test.iloc[:, 0:-self.horizon], self.test.iloc[:, -self.horizon:]

        train_X = train_X.values.reshape(train_X.shape[0], 1, train_X.shape[1])
        val_X = val_X.values.reshape(val_X.shape[0], 1, val_X.shape[1])
        test_X = test_X.values

        train_Y = train_Y.values
        val_Y = val_Y.values
        test_Y = test_Y.values

        return train_X, train_Y, val_X, val_Y, test_X, test_Y

    def get_forecast(self, test_values, model):
        fc_array = []

        for sample in range(0, len(test_values), self.horizon):
            test_nn = test_values[sample]
            test_nn = test_nn.reshape(1, 1, test_nn.shape[0])
            fc = model.predict(test_nn)
            fc = self.scaler.inverse_transform(fc)
            fc_array.extend(fc.tolist()[0])

        fc_df = pd.DataFrame(fc_array, index=self.data[-14 * constants.TEST_DAYS:].index, columns=['fc'])
        fc_df[fc_df < 0] = 0
        return fc_df
