import pandas as pd
from constants import SWIS_POSTCODES

models = {'0': {'name': 'naive', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '1': {'name': 'arima', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '2': {'name': 'conventional_lstm', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '3': {'name': 'conventional_cnn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '4': {'name': 'conventional_tcn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '5': {'name': 'SWIS_APPROACH_A_more_layer_without_norm', 'dir': 'swis_combined_nn_results/approachA',
                'runs': 10},
          '6': {'name': 'SWIS_APPROACH_B', 'dir': 'swis_combined_nn_results/approachB', 'runs': 10},
          }

stat_models = ['arima', 'naive']
combined = ['SWIS_APPROACH_B', 'SWIS_APPROACH_A_more_layer_without_norm']
conventional_nns = ['conventional_lstm', 'conventional_cnn', 'conventional_tcn']


def sum_fc_results(ts_array, model_path, model_name):
    dfs = []
    for ts in ts_array:
        if model_name in stat_models:
            ts_fc = pd.read_csv(f'{model_path}/{model_name}/{ts}.csv', index_col=[0])[['fc']]
        elif model_name in conventional_nns:
            ts_fc = pd.read_csv(f'{model_path}/{model_name}/{ts}/0/grid.csv', index_col=[0])[['fc']]
        else:
            ts_fc = pd.read_csv(f'{model_path}/{model_name}/0/grid.csv', index_col=[0])[['fc']]
        dfs.append(ts_fc)
    concat_df = pd.concat(dfs, axis=1).sum(axis=1)
    return concat_df


all_fc = []
for _, model_info in models.items():
    name = model_info['name']
    path = model_info['dir']
    print(name)

    grid_td = pd.DataFrame(sum_fc_results(['grid'], path, name), columns=[f'{name}_grid'])
    all_fc.append(grid_td)
    if name not in combined:
        # we get two forecasts, grid and pc aggregated
        grid_bu = pd.DataFrame(sum_fc_results(SWIS_POSTCODES, path, name), columns=[f'{name}_pc'])
        all_fc.append(grid_bu)

all_fc_df = pd.concat(all_fc, axis=1)
all_fc_df.to_csv('swis_combined_nn_results/forecasts.csv')


