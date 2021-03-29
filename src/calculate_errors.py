import numpy as np


# calculate the MASE for a given horizon
def MASE(training_series, testing_series, prediction_series, m, denom=None):
    h = testing_series.shape[0]
    numerator = np.sum(np.abs(testing_series - prediction_series))
    # print(numerator)

    n = training_series.shape[0]
    # denominator = np.abs(np.diff(training_series, m)).sum() / (n - m)
    if denom is None:
        ne = 0
        for i in range(m, len(training_series)):
            ne = ne + abs(training_series[i] - training_series[i - m])
        denominator = ne / (n - m)
    else:
        denominator = denom
    # print(denominator)
    return (numerator / denominator) / h


# calculate the MASE for all test samples
def test_errors(train_sample, test_sample, forecasts, horizon, seasonality, denom=None):
    errors = []
    for h in range(0, len(test_sample), horizon):
        # let's add a condition to only calculate the error after sunrise and before sunset
        if np.count_nonzero(test_sample[h: h + horizon]):
            error = MASE(train_sample, test_sample[h: h + horizon], forecasts[h: h + horizon], seasonality, denom)
            # print(error)
            errors.append(error)

    return np.mean(errors), errors


def calculate_errors(df, df_fc, seasonality, horizon):
    mase_vals = {}
    mase_ls = []
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    test_df = df[int(n*0.9):]

    for col in df.columns:
        col_data = train_df[col]
        denominator = calculate_denom(col_data, seasonality)
        y_actual = test_df[col].values
        y_predicted = df_fc[col].values
        mase, mase_arr = test_errors(col_data.values, y_actual, y_predicted, horizon, seasonality, denominator)
        mase_ls.append(mase)
        mase_vals[col] = mase_arr
    return mase_ls, mase_vals


def calculate_denom(train_df, m):
    ne = 0
    n = train_df.values.shape[0]
    for i in range(m, len(train_df.values)):
        ne = ne + abs(train_df.values[i][0] - train_df.values[i - m][0])
        denominator = ne / (n - m)
    return denominator