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
        # clustering approach
        elif model_name in clustering:
            if ts in const.OTHER_TS:
                ts_fc = \
                    pd.read_csv(f'swis_conventional_nn_results/conventional_tcn/{ts}/{run}/grid.csv', index_col=[0])[
                        ['fc']]
            else:
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
    if model_name not in no_grid:
        grid_rmse = calculate_grid_error(grid_model_path, model_path, [const.ALL_SWIS_TS[0]], run, model_name)
        level_rmse.append(grid_rmse)
    if notcombined:

        if model_name in clustering:
            ts_to_run = []
            for clusters in const.CLUSTER_TS:
                ts_to_run.append(f'cluster_{clusters}')
            for other_ts in const.OTHER_TS:
                ts_to_run.append(other_ts)
            cluster_rmse = calculate_grid_error(grid_model_path, model_path, ts_to_run, run, model_name)
            level_rmse.append(cluster_rmse)
        else:
            pc_rmse = calculate_grid_error(grid_model_path, model_path, const.SWIS_POSTCODES, run, model_name)
            level_rmse.append(pc_rmse)
    return level_rmse


# models = {'0': {'name': 'naive', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
#           '1': {'name': 'arima', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
#           '2': {'name': 'conventional_lstm', 'dir': 'swis_conventional_nn_results', 'runs': 10},
#           '3': {'name': 'SWIS_APPROACH_A', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
#           '4': {'name': 'conventional_cnn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
#           '5': {'name': 'conventional_tcn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
#           '6': {'name': 'SWIS_APPROACH_B_with_clustering', 'dir': 'swis_combined_nn_results/approachB', 'runs': 10},
#           '7': {'name': 'SWIS_APPROACH_B', 'dir': 'swis_combined_nn_results/approachB', 'runs': 10},
#           '8': {'name': 'SWIS_APPROACH_A_SKIP', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
#           '9': {'name': 'SWIS_APPROACH_A_more_layer', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
#           '10': {'name': 'SWIS_APPROACH_A_more_layer_without_norm', 'dir': 'swis_combined_nn_results/approachA',
#                  'runs': 10},
#           '11': {'name': 'SWIS_APPROACH_A_more_layer_with_simple_CNN', 'dir': 'swis_combined_nn_results/approachA',
#                  'runs': 10},
#           '12': {'name': 'SWIS_APPROACH_B_with_clustering2', 'dir': 'swis_combined_nn_results/approachB', 'runs': 10},
#           '13': {'name': 'sequentional_training_approach', 'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
#           '14': {'name': 'SWIS_APPROACH_A_SKIP_GRID_SKIP', 'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
#           '15': {'name': 'SWIS_APPROACH_A_more_layer_without_norm_grid_skip',
#                  'dir': 'swis_combined_nn_results/approachA', 'runs': 10},
#           '16': {'name': 'pc_together_2D_conv_approach',
#                  'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
#           '17': {'name': 'pc_together_2D_conv_approach_with_grid',
#                  'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
#           '18': {'name': 'approachA_increase_number_of_filters',
#                  'dir': 'swis_combined_nn_results/approachA', 'runs': 3},
#           '19': {'name': 'pc_together_2D_conv_approach_with_simple_grid_cnn',
#                  'dir': 'swis_combined_nn_results/approachB', 'runs': 3},
#           '20': {'name': 'SWIS_APPROACH_A_reshape_appraoch',
#                  'dir': 'swis_combined_nn_results/approachA', 'runs': 3},
#           '21': {'name': 'SWIS_APPROACH_B_max_pool',
#                  'dir': 'swis_combined_nn_results/approachB', 'runs': [0, 1, 2, 3, 5, 6, 7, 8, 9]},
#           '22': {'name': 'SWIS_APPROACH_B_with_fully_connected',
#                  'dir': 'swis_combined_nn_results/approachB', 'runs': [1, 2, 3, 4, 5, 6, 7, 9]}
#           }

# models = {'0': {'name': 'pc_2d_conv_with_grid_tcn', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '1': {'name': 'pc_2d_conv_with_grid_tcn_method2', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '2': {'name': 'SWIS_APPROACH_A_more_layer_without_norm_grid_skip', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '3': {'name': 'swis_pc_grid_parallel', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10}
#           }

models = {'1': {'name': 'grid_conv_in_each_pc_seperately', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
          '5': {'name': 'concat_pc_with_grid_tcn2', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
          '8': {'name': 'concat_pc_with_grid_tcn3', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
          '21': {'name': 'concat_pc_with_grid_tcn2_for_cluster', 'dir': 'swis_combined_nn_results/new_models',
                 'runs': 10},
          '10': {'name': 'SWIS_APPROACH_A_more_layer_without_norm', 'dir': 'swis_combined_nn_results/approachA',
                 'runs': 10}}

# models = {'0': {'name': 'SWIS_APPROACH_A_with_weather_only', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '1': {'name': 'grid_conv_in_each_pc_seperately', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '2': {'name': 'concat_pc_with_grid_tcn', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '3': {'name': 'pc_2d_conv_with_grid_tcn', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '4': {'name': 'pc_2d_conv_with_grid_tcn_method2', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '5': {'name': 'concat_pc_with_grid_tcn2', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '6': {'name': 'concat_pc_with_grid_tcn2_with_batchnorm', 'dir': 'swis_combined_nn_results/new_models',
#                 'runs': 10},
#           '7': {'name': 'concat_pc_with_grid_tcn2_with_layernorm', 'dir': 'swis_combined_nn_results/new_models',
#                 'runs': 10},
#           '8': {'name': 'concat_pc_with_grid_tcn3', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '9': {'name': 'concat_pc_with_grid_tcn4_lr', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '10': {'name': 'concat_pc_with_grid_tcn4', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '11': {'name': 'concat_pc_with_grid_tcn2_lr', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '12': {'name': 'conv_3d_model', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '13': {'name': 'concat_pc_with_grid_tcn5', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '14': {'name': 'concat_pc_with_grid_tcn6', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '15': {'name': 'conv_3d_model_2', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '16': {'name': 'concat_pc_with_grid_at_each_tcn', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '17': {'name': 'concat_pc_with_grid_tcn2_new', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '18': {'name': 'concat_pc_with_grid_tcn2_relu_and_norm', 'dir': 'swis_combined_nn_results/new_models',
#                  'runs': 10},
#           '19': {'name': 'concat_pc_with_grid_tcn2_lr_decay', 'dir': 'swis_combined_nn_results/new_models', 'runs': 10},
#           '20': {'name': 'concat_pc_with_grid_tcn2_concat_at_end', 'dir': 'swis_combined_nn_results/new_models',
#                  'runs': 10},
#           '21': {'name': 'concat_pc_with_grid_tcn2_for_cluster', 'dir': 'swis_combined_nn_results/new_models',
#                  'runs': 10}

# }
stat_models = ['arima', 'naive']
# combined = ['SWIS_APPROACH_A', 'SWIS_APPROACH_B', 'SWIS_APPROACH_B_with_clustering', 'SWIS_APPROACH_A_SKIP',
#             'SWIS_APPROACH_B_with_clustering2', 'SWIS_APPROACH_A_more_layer', 'SWIS_APPROACH_A_more_layer_without_norm',
#             'SWIS_APPROACH_A_more_layer_with_simple_CNN', 'sequentional_training_approach',
#             'SWIS_APPROACH_A_SKIP_GRID_SKIP', 'SWIS_APPROACH_A_more_layer_without_norm_grid_skip',
#             'pc_together_2D_conv_approach', 'pc_together_2D_conv_approach_with_grid'
#     , 'approachA_increase_number_of_filters', 'pc_together_2D_conv_approach_with_simple_grid_cnn',
#             'SWIS_APPROACH_A_reshape_appraoch', 'SWIS_APPROACH_B_max_pool',
#             'SWIS_APPROACH_B_with_fully_connected']

combined = ['pc_2d_conv_with_grid_tcn', 'pc_2d_conv_with_grid_tcn_method2',
            'SWIS_APPROACH_A_more_layer_without_norm_grid_skip', 'swis_pc_grid_parallel',
            'SWIS_APPROACH_A_with_weather_only', 'concat_pc_with_grid_tcn', 'concat_pc_with_grid_tcn2',
            'concat_pc_with_grid_tcn2_with_batchnorm',
            'concat_pc_with_grid_tcn2_with_layernorm', 'concat_pc_with_grid_tcn3', 'concat_pc_with_grid_tcn4_lr',
            'concat_pc_with_grid_tcn4',
            'concat_pc_with_grid_tcn2_lr', 'conv_3d_model', 'concat_pc_with_grid_tcn5', 'concat_pc_with_grid_tcn6',
            'conv_3d_model_2', 'concat_pc_with_grid_tcn2_new', 'concat_pc_with_grid_at_each_tcn',
            'concat_pc_with_grid_tcn2_relu_and_norm', 'concat_pc_with_grid_tcn2_lr_decay',
            'concat_pc_with_grid_tcn2_concat_at_end', 'SWIS_APPROACH_A']
conventional_nns = ['conventional_lstm', 'conventional_cnn', 'conventional_tcn', 'grid_conv_in_each_pc_seperately',
                    'SWIS_APPROACH_A_more_layer_without_norm']
no_grid = ['grid_conv_in_each_pc_seperately', 'concat_pc_with_grid_tcn2_for_cluster']
clustering = ['concat_pc_with_grid_tcn2_for_cluster']
# model_number = sys.argv[1]
# MODEL_NAME = models[model_number]['name']
# PATH = models[model_number]['dir']
# RUN_RANGE = models[model_number]['runs']

# print(MODEL_NAME)

all_errors = []

for model_number in models:
    MODEL_NAME = models[model_number]['name']
    PATH = models[model_number]['dir']
    RUN_RANGE = models[model_number]['runs']
    print(MODEL_NAME)
    dir_path = f'{PATH}/{MODEL_NAME}'
    one_grid_path = f'{PATH}/{MODEL_NAME}'
    for RUN in range(0, RUN_RANGE):
        if MODEL_NAME in conventional_nns:
            one_grid_path = f'{PATH}/{MODEL_NAME}/grid/{RUN}'
            if MODEL_NAME in no_grid:
                one_grid_path = f'swis_conventional_nn_results/conventional_tcn/grid/{RUN}'
        elif MODEL_NAME in combined:
            one_grid_path = f'{PATH}/{MODEL_NAME}/{RUN}'
        elif MODEL_NAME in clustering:
            one_grid_path = f'swis_conventional_nn_results/conventional_tcn/grid/{RUN}'
        notcombined = True
        if MODEL_NAME in combined:
            notcombined = False
        rmse_run_list = get_grid_error_per_run(one_grid_path, dir_path, RUN, MODEL_NAME, notcombined)

        if MODEL_NAME not in no_grid:
            all_errors.append([MODEL_NAME, rmse_run_list[0], RUN, 'grid'])
        else:
            all_errors.append([MODEL_NAME, rmse_run_list[0], RUN, 'pc'])
        if notcombined and (MODEL_NAME not in no_grid):
            all_errors.append([MODEL_NAME, rmse_run_list[1], RUN, 'pc'])

all_error_df = pd.DataFrame(all_errors, columns=['model_name', 'error', 'run', 'Level'])

error_path = f'swis_combined_nn_results/all_fc_errors.csv'
all_error_df.to_csv(error_path)
# if not os.path.exists(error_path):
#     os.makedirs(error_path)
# all_error_df.to_csv(f'{error_path}/final_errors.csv')

# mean_std_df = all_error_df.groupby(by='Level').agg({'NRMSE': ['mean', 'std']})
# mean_err = mean_std_df.values[0][0]
# std_err = mean_std_df.values[0][1]
#
# # read the current error file -- note: I am currently running the combined approaches so this will be okay
# FILE = 'swis_combined_nn_results/errors.csv'
# errors = pd.read_csv(FILE, index_col=0)
# errors = errors.append({'Name': MODEL_NAME, 'mean NRMSE': mean_err, 'std': std_err}, ignore_index=True)
# errors = errors.sort_values(by='mean NRMSE')
# errors.to_csv(FILE)
