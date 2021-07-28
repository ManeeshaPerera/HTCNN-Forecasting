from constants import random_5_samples, random_25_samples, random_50_samples, random_75_samples, random_sample_101
import pandas as pd
import src.calculate_errors as err
import src.utils as util

models = {'0': {'name': 'naive', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '1': {'name': 'arima', 'dir': 'benchmark_results/swis_benchmarks', 'runs': 1},
          '2': {'name': 'conventional_lstm', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '4': {'name': 'conventional_cnn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          '5': {'name': 'conventional_tcn', 'dir': 'swis_conventional_nn_results', 'runs': 10},
          }
stat_models = ['arima', 'naive']
conventional_nns = ['conventional_lstm', 'conventional_cnn', 'conventional_tcn']
all_methods = ['arima', 'naive', 'conventional_lstm', 'conventional_cnn', 'conventional_tcn']
# sample_list = [random_5_samples, random_25_samples, random_50_samples, random_75_samples, random_sample_101]
sample_list = [random_5_samples, random_25_samples, random_50_samples, random_75_samples]
data_frame_list = []


def calculate_error(pc_list_func, model_name, model_path, run_for_fun):
    pc_fcs = []
    pc_power = []
    gird_level = []
    for pc in pc_list_func:
        pc_all_data = pd.read_csv(f'swis_ts_data/ts_data/{pc}.csv', index_col=0)
        if model_name in stat_models:
            data_file = pd.read_csv(f'{model_path}/{pc}.csv', index_col=[0])
        else:
            # conventional nns
            data_file = pd.read_csv(f'{model_path}/{pc}/{run_for_fun}/grid.csv', index_col=[0])
        gird_level.append(pc_all_data[['power']])
        pc_fcs.append(data_file[['fc']])
        pc_power.append(data_file[['power']])
    pc_fc_df = pd.concat(pc_fcs, axis=1).sum(axis=1)
    pc_power_df = pd.concat(pc_power, axis=1).sum(axis=1)
    pc_all_data_df = pd.concat(gird_level, axis=1).sum(axis=1)
    train, _ = util.split_train_test_statmodels_swis(pc_all_data_df)
    mean_err, _ = err.test_errors_nrmse(train.values, pc_power_df.values, pc_fc_df.values, 18)
    return mean_err


# category_num = 0
# for category in sample_list:
#     print("random_sample_category", category_num)
#     pc_combination = 0
#     for pc_list in category:
#         print("pc random in that sample", pc_combination)
#         print(pc_list)
#         gird_level = []
#         for pc in pc_list:
#             pc_all_data = pd.read_csv(f'swis_ts_data/ts_data/{pc}.csv', index_col=0)
#             gird_level.append(pc_all_data.iloc[:, 0:7])
#         pc_all_data_df = pd.concat(gird_level, axis=1)
#         print(pc_all_data_df)
#         df = pc_all_data_df.groupby(level=0, axis=1).sum()
#         df = df.reindex(['power','day2', 'day3', 'day4', 'day5', 'day6', 'day7'], axis=1).round(2)
#         print(df)
#         df.to_csv(f'swis_ts_data/category_{category_num}/grid_sample_{pc_combination}.csv')
#         # for key, method in models.items():
#         #     name = method['name']
#         #     run = method['runs']
#         #     directory = method['dir']
#         #     path = f'{directory}/{name}'
#         #     for run_val in range(0, run):
#         #         mean_error = calculate_error(pc_list, name, path, run_val)
#         #         data_frame_list.append([run_val, pc_combination, category_num, name, mean_error])
#         pc_combination += 1
#     category_num += 1

# data_frame_store = pd.DataFrame(data_frame_list, columns=['run', 'sample', 'category', 'method', 'error'])
# data_frame_store.to_csv('swis_ts_data/hf_dataframe_errors.csv')

def calculate_our_method_error(fc_dataframe, category_no, sample_num):
    grid_values = pd.read_csv(f'swis_ts_data/category_{category_no}/grid_sample_{sample_num}.csv', index_col=0)
    train, _ = util.split_train_test_statmodels_swis(grid_values)
    pc_power_df = fc_dataframe[['power']]
    pc_fc_df = fc_dataframe[['fc']]
    mean_err, _ = err.test_errors_nrmse(train.values, pc_power_df.values, pc_fc_df.values, 18)
    return mean_err


category_num = 0
for category in sample_list[0:1]:
    print("random_sample_category", category_num)
    pc_combination = 0
    for pc_list in category:
        print("pc random in that sample", pc_combination)
        print(pc_list)
        gird_level = []
        name = 'method-AA'
        run = 10
        for run_val in range(0, run):
            directory = f'swis_ts_data/category_{category_num}/{name}/{run_val}'
            # print(directory)
            try:
                sample_fc = pd.read_csv(f'{directory}/grid_sample_{pc_combination}.csv', index_col=0)
                mean_error = calculate_our_method_error(sample_fc, category_num, pc_combination)
                data_frame_list.append([run_val, pc_combination, category_num, name, mean_error])
            except:
                print("no file")
        pc_combination += 1
    category_num += 1

data_frame_store = pd.DataFrame(data_frame_list, columns=['run', 'sample', 'category', 'method', 'error'])
data_frame_store.to_csv('swis_ts_data/hf_dataframe_errors_methodAA.csv')