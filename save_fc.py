import pandas as pd
import constants as const

def sum_fc_results(ts_array, model_path, run, model_name):
    dfs = []
    for ts in ts_array:
        if model_name in stat_models:
            ts_fc = pd.read_csv(f'{model_path}/{ts}.csv', index_col=[0])[['fc']]
        elif model_name in conventional_nns:
            ts_fc = pd.read_csv(f'{model_path}/{ts}/{run}/grid.csv', index_col=[0])[['fc']]
        # clustering approach
        elif model_name in clustering or model_name in clustering_and_pc:
            if ts in const.OTHER_TS:
                ts_fc = \
                    pd.read_csv(f'swis_conventional_nn_results/conventional_tcn/{ts}/{run}/grid.csv', index_col=[0])[
                        ['fc']]
            else:
                ts_fc = pd.read_csv(f'{model_path}/{ts}/{run}/grid.csv', index_col=[0])[['fc']]
        else:
            ts_fc = pd.read_csv(f'{model_path}/{run}/grid.csv', index_col=[0])[['fc']]
        dfs.append(ts_fc)
    concat_df = pd.DataFrame(pd.concat(dfs, axis=1).sum(axis=1), columns=[model_name])
    return concat_df


def get_grid_error_per_run(grid_model_path, model_path, run, model_name, notcombined=True):
    if model_name not in no_grid:
        grid_fc= sum_fc_results([const.ALL_SWIS_TS[0]], model_path, run, model_name)
        print(model_name, grid_fc)
        return grid_fc
    if notcombined:

        if model_name in clustering:
            ts_to_run = []
            for clusters in const.CLUSTER_TS:
                ts_to_run.append(f'cluster_{clusters}')
            for other_ts in const.OTHER_TS:
                ts_to_run.append(other_ts)
            cluster_fc = sum_fc_results(ts_to_run, model_path, run, model_name)
            print(model_name, cluster_fc)
            return cluster_fc
        else:
            pc_fc= sum_fc_results(const.SWIS_POSTCODES, model_path, run, model_name)
            print(model_name, pc_fc)
            return pc_fc



models = {'0': {'name': 'naive', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '1': {'name': 'arima', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '2': {'name': 'conventional_lstm', 'dir': 'swis_conventional_nn_results', 'runs': 1},
          '3': {'name': 'conventional_cnn', 'dir': 'swis_conventional_nn_results', 'runs': 1},
          '4': {'name': 'conventional_tcn', 'dir': 'swis_conventional_nn_results', 'runs': 1},
          '5': {'name': 'concat_pc_with_grid_tcn2', 'dir': 'swis_combined_nn_results/new_models', 'runs': 1},
          '21': {'name': 'concat_pc_with_grid_tcn2_for_cluster', 'dir': 'swis_combined_nn_results/new_models',
                 'runs': 1},
          '10': {'name': 'SWIS_APPROACH_A_more_layer_without_norm', 'dir': 'swis_combined_nn_results/approachA',
                 'runs': 1},
          '11': {'name': 'SWIS_APPROACH_A_more_layer_without_norm_cluster',
                 'dir': 'swis_combined_nn_results/new_models',
                 'runs': 1},
          '14': {'name': 'conventional_TCN_approach',
                 'dir': 'swis_combined_nn_results/new_models',
                 'runs': 1}
          }

stat_models = ['arima', 'naive']

combined = ['pc_2d_conv_with_grid_tcn', 'pc_2d_conv_with_grid_tcn_method2',
            'SWIS_APPROACH_A_more_layer_without_norm_grid_skip', 'swis_pc_grid_parallel',
            'SWIS_APPROACH_A_with_weather_only', 'concat_pc_with_grid_tcn', 'concat_pc_with_grid_tcn2',
            'concat_pc_with_grid_tcn2_with_batchnorm',
            'concat_pc_with_grid_tcn2_with_layernorm', 'concat_pc_with_grid_tcn3', 'concat_pc_with_grid_tcn4_lr',
            'concat_pc_with_grid_tcn4',
            'concat_pc_with_grid_tcn2_lr', 'conv_3d_model', 'concat_pc_with_grid_tcn5', 'concat_pc_with_grid_tcn6',
            'conv_3d_model_2', 'concat_pc_with_grid_tcn2_new', 'concat_pc_with_grid_at_each_tcn',
            'concat_pc_with_grid_tcn2_relu_and_norm', 'concat_pc_with_grid_tcn2_lr_decay',
            'concat_pc_with_grid_tcn2_concat_at_end', 'SWIS_APPROACH_A', 'SWIS_APPROACH_A_more_layer_without_norm',
            'SWIS_APPROACH_A_more_layer_without_norm_weather_centroids', 'concat_pc_with_grid_tcn2_weather_centroids']
conventional_nns = ['conventional_lstm', 'conventional_cnn', 'conventional_tcn', 'grid_conv_in_each_pc_seperately']
no_grid = ['grid_conv_in_each_pc_seperately', 'concat_pc_with_grid_tcn2_for_cluster',
           'SWIS_APPROACH_A_more_layer_without_norm_cluster', 'conventional_TCN_approach']
clustering = ['concat_pc_with_grid_tcn2_for_cluster', 'SWIS_APPROACH_A_more_layer_without_norm_cluster']
clustering_and_pc = ['conventional_TCN_approach']

all_fc = []

for model_number in models:
    MODEL_NAME = models[model_number]['name']
    PATH = models[model_number]['dir']
    RUN_RANGE = models[model_number]['runs']
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
        elif MODEL_NAME in clustering_and_pc:
            one_grid_path = f'swis_conventional_nn_results/conventional_tcn/grid/{RUN}'
        notcombined = True
        if MODEL_NAME in combined:
            notcombined = False
        fc_df = get_grid_error_per_run(one_grid_path, dir_path, RUN, MODEL_NAME, notcombined)
        all_fc.append(fc_df)
print(pd.concat(all_fc, axis=1))