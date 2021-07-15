import constants as const
import pandas as pd
import src.utils as util
import src.calculate_errors as err
import sys
import os


def sum_fc_results(ts_array, model_path, run, model_name):
    dfs = []
    for ts in ts_array:
        if model_name in stat_models:
            ts_fc = pd.read_csv(f'{model_path}/{ts}.csv', index_col=[0])[['fc']]
        elif model_name in conventional_nns:
            ts_fc = pd.read_csv(f'{model_path}/{ts}/{run}/grid.csv', index_col=[0])[['fc']]
        else:
            ts_fc = pd.read_csv(f'{model_path}/{run}/grid.csv', index_col=[0])[['fc']]
        dfs.append(ts_fc)
    concat_df = pd.concat(dfs, axis=1).sum(axis=1)
    return concat_df


def calculate_grid_error(grid_model_path, dir_path, ts_array, run, model_name):
    data = pd.read_csv('swis_ts_data/ts_data/grid.csv', index_col=[0])

    train, test = util.split_train_test_statmodels_swis(data)
    train_df = train[['power']]

    # let's take the power values from one dataframe
    results_df = pd.read_csv(f'{grid_model_path}/grid.csv', index_col=[0])
    test_sample = results_df['power'].values

    forecasts = sum_fc_results(ts_array, dir_path, run, model_name).values
    horizon = 18

    mean_err, _ = err.test_errors_nrmse(train_df.values, test_sample, forecasts, horizon)
    return mean_err


def get_grid_error_per_run(grid_model_path, model_path, run, model_name, notcombined=True):
    level_rmse = []
    grid_rmse = calculate_grid_error(grid_model_path, model_path, [const.ALL_SWIS_TS[0]], run, model_name)

    level_rmse.append(grid_rmse)
    if notcombined:
        pc_rmse = calculate_grid_error(grid_model_path, model_path, const.SWIS_POSTCODES, run, model_name)
        level_rmse.append(pc_rmse)
    return level_rmse


models = {'0': {'name': 'naive', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '1': {'name': 'arima', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '2': {'name': 'conventional_lstm', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '3': {'name': 'SWIS_APPROACH_A', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
          '4': {'name': 'conventional_cnn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '5': {'name': 'conventional_tcn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '6': {'name': 'SWIS_APPROACH_B_with_clustering', 'dir': 'swis_combined_nn_results/approachB', 'runs': 1},
          '7': {'name': 'SWIS_APPROACH_B', 'dir': 'swis_combined_nn_results/approachB', 'runs': 1},
          '8': {'name': 'SWIS_APPROACH_A_SKIP', 'dir': 'swis_combined_nn_results/approachA', 'runs': 1},
          '9': {'name': 'SWIS_APPROACH_B_with_fully_connected', 'dir': 'swis_combined_nn_results/approachB', 'runs': 1}}

stat_models = ['arima', 'naive']
combined = ['SWIS_APPROACH_A', 'SWIS_APPROACH_B', 'SWIS_APPROACH_B_with_clustering', 'SWIS_APPROACH_A_SKIP',
            'SWIS_APPROACH_B_with_fully_connected']
conventional_nns = ['conventional_lstm', 'conventional_cnn', 'conventional_tcn']
model_number = sys.argv[1]
MODEL_NAME = models[model_number]['name']
PATH = models[model_number]['dir']
RUN_RANGE = models[model_number]['runs']

print(MODEL_NAME)

all_errors = []
dir_path = f'{PATH}/{MODEL_NAME}'
one_grid_path = f'{PATH}/{MODEL_NAME}'

for RUN in range(0, RUN_RANGE):
    if MODEL_NAME in conventional_nns:
        one_grid_path = f'{PATH}/{MODEL_NAME}/grid/{RUN}'
    elif MODEL_NAME in combined:
        one_grid_path = f'{PATH}/{MODEL_NAME}/{RUN}'
    notcombined = True
    if MODEL_NAME in combined:
        notcombined = False
    rmse_run_list = get_grid_error_per_run(one_grid_path, dir_path, RUN, MODEL_NAME, notcombined)
    all_errors.append([rmse_run_list[0], 'grid'])
    if notcombined:
        all_errors.append([rmse_run_list[1], 'pc'])

all_error_df = pd.DataFrame(all_errors, columns=['NRMSE', 'Level'])

error_path = f'{dir_path}/errors/'
if not os.path.exists(error_path):
    os.makedirs(error_path)
all_error_df.to_csv(f'{error_path}/final_errors.csv')
