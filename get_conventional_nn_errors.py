import constants as const
import pandas as pd
import src.utils as util
import src.calculate_errors as err
import sys


def sum_fc_results(ts_array, model_path, run):
    dfs = []
    for ts in ts_array:
        ts_fc = pd.read_csv(f'{model_path}/{ts}/{run}/grid.csv', index_col=[0])[['fc']]
        dfs.append(ts_fc)
    concat_df = pd.concat(dfs, axis=1).sum(axis=1)
    return concat_df


def calculate_grid_error(start, end, grid_model_path, dir_path, run):
    if end == -1:
        ts_array = const.TS[start:]
    else:
        ts_array = const.TS[start: end]
    data = pd.read_csv('ts_data/grid.csv', index_col=[0])
    look_back = 14 * 7  # 14 hours in to 7 days

    # train, val, test split
    train, val, test = util.split_hourly_data(data, look_back)
    train_df = train[['power']]

    # let's take the power values from one dataframe
    results_df = pd.read_csv(f'{grid_model_path}/grid.csv', index_col=[0])
    test_sample = results_df['power'].values

    forecasts = sum_fc_results(ts_array, dir_path, run).values
    horizon = 14

    mean_err, _ = err.test_errors_nrmse(train_df.values, test_sample, forecasts, horizon)
    return mean_err


def get_grid_error_per_run(grid_model_path, model_path, run):
    level_rmse = []
    grid_rmse = calculate_grid_error(0, 1, grid_model_path, model_path, run)
    tl_rmse = calculate_grid_error(1, 3, grid_model_path, model_path, run)
    sub_rmse = calculate_grid_error(3, 7, grid_model_path, model_path, run)
    pc_rmse = calculate_grid_error(7, 13, grid_model_path, model_path, run)
    site_rmse = calculate_grid_error(13, -1, grid_model_path, model_path, run)

    level_rmse.append(grid_rmse)
    level_rmse.append(tl_rmse)
    level_rmse.append(sub_rmse)
    level_rmse.append(pc_rmse)
    level_rmse.append(site_rmse)

    return level_rmse


models = {'0': 'conventional_lstm', '1': 'conventional_cnn', '2': 'conventional_tcn'}
model_number = sys.argv[1]
model_name = models[model_number]

print(model_name)

all_errors = []
dir_path = f'{model_name}'
# one grid path
one_grid_path = f'{model_name}/grid'

for run in range(0, 10):
    rmse_run_list = get_grid_error_per_run(one_grid_path, dir_path, run)
    all_errors.append([rmse_run_list[0], 'grid'])
    all_errors.append([rmse_run_list[1], 'tl'])
    all_errors.append([rmse_run_list[2], 'substation'])
    all_errors.append([rmse_run_list[3], 'pc'])
    all_errors.append([rmse_run_list[4], 'site'])

all_error_df = pd.DataFrame(all_errors, columns=['NRMSE', 'Level'])
all_error_df.to_csv(f'{dir_path}/final_errors.csv')
