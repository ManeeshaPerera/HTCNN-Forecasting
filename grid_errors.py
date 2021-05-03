import pandas as pd
import constants as const
import src.utils as util
import src.calculate_errors as err


def sum_fc_results(ts_array, model_path):
    dfs = []
    for ts in ts_array:
        ts_fc = pd.read_csv(f'{model_path}/final_results/{ts}.csv', index_col=[0])[['average_fc']]
        dfs.append(ts_fc)
    concat_df = pd.concat(dfs, axis=1).sum(axis=1)
    return concat_df


def calculate_grid_error(start, end, model_path1, model_path2, error_metric):
    if end == -1:
        ts_array = const.TS[start:]
    else:
        ts_array = const.TS[start: end]
    data = pd.read_csv('ts_data/grid.csv', index_col=[0])
    look_back = 14 * 7  # 14 hours in to 7 days

    # train, val, test split
    train, val, test = util.split_hourly_data(data, look_back)
    train_df = train[['power']]
    denom = err.calculate_denom(train_df, const.H_SEASONALITY)

    results_df = pd.read_csv(f'{model_path1}/final_results/grid.csv', index_col=[0])
    test_sample = results_df['power'].values
    forecasts = sum_fc_results(ts_array, model_path2).values
    horizon = 14

    if error_metric == "RMSE":
        mean_err, error_dist = err.test_errors_nrmse(train_df.values, test_sample, forecasts, horizon)
    else:
        mean_err, error_dist = err.test_errors(train_df, test_sample, forecasts, horizon, const.H_SEASONALITY, denom)

    return mean_err, error_dist


def mase_grid(model_path1, model_path2, error_metric="MASE"):
    level_mase = []
    level_mase_dist = {}
    grid_mase, grid_dist = calculate_grid_error(0, 1, model_path1, model_path1, error_metric)
    tl_mase, tl_dist = calculate_grid_error(1, 3, model_path1, model_path1, error_metric)
    sub_mase, sub_dist = calculate_grid_error(3, 7, model_path1, model_path1, error_metric)
    pc_mase, pc_dist = calculate_grid_error(7, 13, model_path1, model_path2, error_metric)
    site_mase, site_dist = calculate_grid_error(13, -1, model_path1, model_path2, error_metric)

    level_mase.append(grid_mase)
    level_mase.append(tl_mase)
    level_mase.append(sub_mase)
    level_mase.append(pc_mase)
    level_mase.append(site_mase)

    level_mase_dist['grid'] = grid_dist
    level_mase_dist['tl'] = tl_dist
    level_mase_dist['sub'] = sub_dist
    level_mase_dist['pc'] = pc_dist
    level_mase_dist['site'] = site_dist

    return level_mase, level_mase_dist

def get_df_method(error_list, method):
    df = pd.DataFrame(error_list).transpose()
    df.columns = ['Grid', 'TL - Aggregated', 'SUB - Aggregated', 'PC - Aggregated', 'Site - Aggregated']
    df.index = [method]
    return df


if __name__ == '__main__':
    MASE = []
    RMSE = []

    cell_dims = [32, 64]
    num_layers2 = [3, 5]
    num_layers = [2, 3]

    learning_rate = [0.001, 0.0001]
    epochs = 1000

    lookback = [1, 3, 7]
    for cell_dim in cell_dims:
        for layer in range(0, len(num_layers)):
            for lr in learning_rate:
                for lag in lookback:
                    lstm_layer1 = num_layers[layer]
                    lstm_layer2 = num_layers2[layer]
                    print(cell_dim, lstm_layer1, lr, lag)
                    print(cell_dim, lstm_layer2, lr, lag)
                    model_name1 = f'{cell_dim}_{lstm_layer1}_{lr}_{lag}'
                    model_name2 = f'{cell_dim}_{lstm_layer2}_{lr}_{lag}'

                    model_dir1 = f'lstm_new_results/lstm_{model_name1}'
                    model_dir2 = f'lstm_new_results/lstm_{model_name2}'
                    mase, _ = mase_grid(model_dir1, model_dir2)
                    rmse, _ = mase_grid(model_dir1, model_dir2, error_metric="RMSE")

                    mase_df = get_df_method(mase, model_name1)
                    rmse_df = get_df_method(rmse, model_name2)

                    MASE.append(mase_df)
                    RMSE.append(rmse_df)
    mase_final_results = pd.concat(MASE).round(3)
    rmse_final_results = pd.concat(RMSE).round(3)

    mase_final_results.to_csv('lstm_new_results/mase.csv')
    rmse_final_results.to_csv('lstm_new_results/rmse.csv')

