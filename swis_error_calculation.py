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
          '6': {'name': 'SWIS_APPROACH_B_with_clustering', 'dir': 'swis_combined_nn_results/approachB', 'runs': 10},
          '7': {'name': 'SWIS_APPROACH_B', 'dir': 'swis_combined_nn_results/approachB', 'runs': 10},
          '8': {'name': 'SWIS_APPROACH_A_SKIP', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
          '9': {'name': 'SWIS_APPROACH_A_more_layer', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
          '10': {'name': 'SWIS_APPROACH_A_more_layer_without_norm', 'dir': 'swis_combined_nn_results/approachA',
                 'runs': 10},
          '11': {'name': 'SWIS_APPROACH_A_more_layer_with_simple_CNN', 'dir': 'swis_combined_nn_results/approachA',
                 'runs': 10},
          '12': {'name': 'SWIS_APPROACH_B_with_clustering2', 'dir': 'swis_combined_nn_results/approachB', 'runs': 10},
          '13': {'name': 'sequentional_training_approach', 'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
          '14': {'name': 'SWIS_APPROACH_A_SKIP_GRID_SKIP', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
          '15': {'name': 'SWIS_APPROACH_A_more_layer_without_norm_grid_skip',
                 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
          '16': {'name': 'pc_together_2D_conv_approach',
                 'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
          '17': {'name': 'pc_together_2D_conv_approach_with_grid',
                 'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
          '18': {'name': 'approachA_increase_number_of_filters',
                 'dir': 'swis_combined_nn_results/approachA', 'runs': 3},
          '19': {'name': 'pc_together_2D_conv_approach_with_simple_grid_cnn',
                 'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
          '20': {'name': 'SWIS_APPROACH_A_reshape_appraoch',
                 'dir': 'swis_combined_nn_results/approachA', 'runs': 3}
          }

stat_models = ['arima', 'naive']
combined = ['SWIS_APPROACH_A', 'SWIS_APPROACH_B', 'SWIS_APPROACH_B_with_clustering', 'SWIS_APPROACH_A_SKIP',
            'SWIS_APPROACH_B_with_clustering2', 'SWIS_APPROACH_A_more_layer', 'SWIS_APPROACH_A_more_layer_without_norm',
            'SWIS_APPROACH_A_more_layer_with_simple_CNN', 'sequentional_training_approach',
            'SWIS_APPROACH_A_SKIP_GRID_SKIP', 'SWIS_APPROACH_A_more_layer_without_norm_grid_skip', 'pc_together_2D_conv_approach', 'pc_together_2D_conv_approach_with_grid'
            , 'approachA_increase_number_of_filters', 'pc_together_2D_conv_approach_with_simple_grid_cnn', 'SWIS_APPROACH_A_reshape_appraoch']
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

mean_std_df = all_error_df.groupby(by='Level').agg({'NRMSE': ['mean', 'std']})
mean_err = mean_std_df.values[0][0]
std_err = mean_std_df.values[0][1]

# read the current error file -- note: I am currently running the combined approaches so this will be okay
FILE = 'swis_combined_nn_results/errors.csv'
errors = pd.read_csv(FILE, index_col=0)
errors.append([MODEL_NAME, mean_err, std_err])
errors = errors.sort_values(by='mean NRMSE')
errors.to_csv(FILE)

