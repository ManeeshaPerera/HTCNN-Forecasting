import pandas as pd
from constants import random_5_samples, random_25_samples, random_50_samples, random_75_samples, random_sample_101
import src.utils as util
import src.calculate_errors as err

sample_list = [random_5_samples, random_25_samples, random_50_samples, random_75_samples, random_sample_101]
data_frame_list = []


def sum_fc_results(ts_array, run):
    dfs = []
    for ts in ts_array:
        ts_fc = pd.read_csv(f'swis_conventional_nn_results/{ts}/{run}/grid.csv', index_col=[0])[['fc']]
        dfs.append(ts_fc)
    concat_df = pd.DataFrame(pd.concat(dfs, axis=1).sum(axis=1), columns=['tcn_fc'])
    return concat_df


def calculate_our_method_error(pc_fc_df, category_no, sample_num, pc_power_df, category_val):
    if category_val == 4:
        grid_values = pd.read_csv(f'swis_ts_data/ts_data/grid.csv.csv', index_col=0)
    else:
        grid_values = pd.read_csv(f'swis_ts_data/category_{category_no}/grid_sample_{sample_num}.csv', index_col=0)
    train, _ = util.split_train_test_statmodels_swis(grid_values)
    mean_err, _ = err.test_errors_nrmse(train.values, pc_power_df.values, pc_fc_df.values, 18)
    return mean_err


category_num = 0
for category in sample_list:
    print("random_sample_category", category_num)
    pc_combination = 0
    for pc_list in category:
        print("pc random in that sample", pc_combination)
        print(pc_list)
        gird_level = []
        name = 'method-A'
        run = 10
        for run_val in range(0, run):
            if category_num == 4:
                directory = 'swis_combined_nn_results/approachA/SWIS_APPROACH_A_more_layer_without_norm_grid_skip'
            else:
                directory = f'swis_ts_data/category_{category_num}/{name}/{run_val}'
            # print(directory)
            if category_num == 4:
                # method A forecast
                sample_fc = pd.read_csv(f'{directory}/grid.csv', index_col=0)[['fc']]
                power = pd.read_csv(f'{directory}/grid.csv', index_col=0)[['power']]
            else:
                # method A forecast
                sample_fc = pd.read_csv(f'{directory}/grid_sample_{pc_combination}.csv', index_col=0)[['fc']]
                power = pd.read_csv(f'{directory}/grid_sample_{pc_combination}.csv', index_col=0)[['power']]

            # let's get the tcn fc and get the mean
            tcn_fc = sum_fc_results(pc_list, run_val)
            fc_df = pd.DataFrame(pd.concat([sample_fc, tcn_fc], axis=1).mean(axis=1), columns=['fc'])

            mean_error = calculate_our_method_error(fc_df, category_num, pc_combination, power, category_num)
            data_frame_list.append([run_val, pc_combination, category_num, 'ensemble', mean_error])
        pc_combination += 1
    category_num += 1

data_frame_store = pd.DataFrame(data_frame_list, columns=['run', 'sample', 'category', 'method', 'error'])
data_frame_store.to_csv('swis_ts_data/hf_dataframe_errors_ensemble.csv')
