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
    # let's take the power values from one dataframe
    results_df = pd.read_csv(f'{grid_model_path}/grid.csv', index_col=[0])
    test_sample = results_df['power'].values

    forecasts = sum_fc_results(ts_array, dir_path, run, model_name).values
    horizon = 18

    mean_err, median_err, _ = err.smape_test_sample(test_sample, forecasts, horizon)
    return mean_err, median_err


def get_grid_error_per_run(grid_model_path, model_path, run, model_name, notcombined=True):
    level_smape = []
    level_smape_median = []
    grid_smape, grid_smape_median = calculate_grid_error(grid_model_path, model_path, [const.ALL_SWIS_TS[0]], run,
                                                         model_name)

    level_smape.append(grid_smape)
    level_smape_median.append(grid_smape_median)
    if notcombined:
        pc_sampe, pc_sampe_meidan = calculate_grid_error(grid_model_path, model_path, const.SWIS_POSTCODES, run,
                                                         model_name)
        level_smape.append(pc_sampe)
        level_smape_median.append(pc_sampe_meidan)
    return level_smape, level_smape_median


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
                 'dir': 'swis_combined_nn_results/approachA', 'runs': 10}
          }

stat_models = ['arima', 'naive']
combined = ['SWIS_APPROACH_A', 'SWIS_APPROACH_B', 'SWIS_APPROACH_B_with_clustering', 'SWIS_APPROACH_A_SKIP',
            'SWIS_APPROACH_B_with_clustering2', 'SWIS_APPROACH_A_more_layer', 'SWIS_APPROACH_A_more_layer_without_norm',
            'SWIS_APPROACH_A_more_layer_with_simple_CNN', 'sequentional_training_approach',
            'SWIS_APPROACH_A_SKIP_GRID_SKIP', 'SWIS_APPROACH_A_more_layer_without_norm_grid_skip']
conventional_nns = ['conventional_lstm', 'conventional_cnn', 'conventional_tcn']
model_number = sys.argv[1]
MODEL_NAME = models[model_number]['name']
PATH = models[model_number]['dir']
RUN_RANGE = models[model_number]['runs']

print(MODEL_NAME)

all_errors = []
all_errors_median = []
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
    smape_mean_run_list, smape_median_run_list = get_grid_error_per_run(one_grid_path, dir_path, RUN, MODEL_NAME,
                                                                        notcombined)
    all_errors.append([smape_mean_run_list[0], 'grid'])
    all_errors_median.append([smape_median_run_list[0], 'grid'])
    if notcombined:
        all_errors.append([smape_mean_run_list[1], 'pc'])
        all_errors_median.append([smape_median_run_list[1], 'pc'])

all_error_df = pd.DataFrame(all_errors, columns=['sMAPE_MEAN', 'Level'])
all_errors_median_df = pd.DataFrame(all_errors_median, columns=['sMAPE_Median', 'Level'])


error_path = f'{dir_path}/smape_errors/'
if not os.path.exists(error_path):
    os.makedirs(error_path)
all_error_df.to_csv(f'{error_path}/smape_mean.csv')
all_errors_median_df.to_csv(f'{error_path}/smape_median.csv')

final_results = f'swis_combined_nn_results/SMAPE'
if not os.path.exists(final_results):
    os.makedirs(final_results)

mean_std_df = all_error_df.groupby(by='Level').agg({'sMAPE_MEAN': ['mean', 'std']})
# we are running the combined results - so I am storing it in that path for ease
mean_std_df.to_csv(f'{final_results}/{MODEL_NAME}_smape_mean.csv')


mean_std_df = all_errors_median_df.groupby(by='Level').agg({'sMAPE_Median': ['mean', 'std']})
# we are running the combined results - so I am storing it in that path for ease
mean_std_df.to_csv(f'{final_results}/{MODEL_NAME}_smape_median.csv')
