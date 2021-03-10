import pandas as pd
import pickle


def post_process_data(model_output, df, horizon, col_name):
    fc_ls = []
    for batch in model_output:
        batch_data = batch[::horizon]
        for h in batch_data:
            fc_ls.extend(h.tolist())
    data = df[col_name]
    n = len(data)
    train_df = data[0:int(n * 0.7)]

    train_mean = train_df.mean()
    train_std = train_df.std()

    test_index = data[int(n * 0.9):].index
    test_len = len(test_index)
    fc_df = pd.DataFrame(fc_ls[-test_len:], index=test_index, columns=[col_name])
    # de normalise the data
    fc_df = (fc_df * train_std) + train_mean
    # replace any negative values with 0
    fc_df[fc_df < 0] = 0
    return fc_df


def load_pickle(file):
    with open(f'../fc_results/{file}', 'rb') as f:
        content = pickle.load(f)
    return content


def combine_hf_fc(df, horizon):
    fc_dfs = []
    for col in df.columns:
        fc_file = load_pickle(f'forecast_{str(col)}')
        fc_dataframe = post_process_data(fc_file, df, horizon, col)
        fc_dfs.append(fc_dataframe)
    return pd.concat(fc_dfs, axis=1)