import src.utils as util
import src.calculate_errors as err
import pandas as pd


def calculate_grid_error(model_path):
    data = pd.read_csv(f'ts_data/grid.csv', index_col=[0])
    look_back = 14 * 7  # 14 hours in to 7 days

    # train, val, test split
    train, val, test = util.split_hourly_data(data, look_back)
    train_df = train[['power']]

    results_df = pd.read_csv(f'{model_path}/grid.csv', index_col=[0])
    test_sample = results_df['power'].values
    forecasts = results_df['fc'].values
    horizon = 14

    mean_err, _ = err.test_errors_nrmse(train_df.values, test_sample, forecasts, horizon)
    return mean_err


def get_multiple_runs_error(model_dir_path):
    run_errors = []
    for run in range(0, 10):
        model_path = f'{model_dir_path}/{run}'
        model_nrmse = calculate_grid_error(model_path)
        run_errors.append(model_nrmse)
    error_df = pd.DataFrame(run_errors, columns=['nrmse'])
    error_df.to_csv(f'{model_dir_path}/errors.csv')
    model_mean = error_df.mean().to_list()[0]
    model_std = error_df.std().to_list()[0]
    model_min = error_df.min().to_list()[0]
    model_max = error_df.max().to_list()[0]
    model_median = error_df.median().to_list()[0]
    return [model_mean, model_std, model_min, model_max, model_median]


# final_test_models = {'0': 'postcode_only_TCN',
#                      '1': 'last_residual_approach_with_TCN',
#                      '2': 'local_and_global_conv_approach_with_TCN',
#                      '3': 'local_conv_with_grid_with_TCN_approach',
#                      '4': 'local_conv_with_grid_conv_TCN_approach',
#                      '5': 'pc_and_grid_input_together',
#                      '6': 'grid_added_at_each_TCN_together',
#                      '7': 'grid_conv_added_at_each_TCN_together',
#                      '8': 'frozen_branch_approach_TCN'
#                      }

final_test_models = {'0': 'possibility_2_ApproachA',
                     '1': 'possibility_3_ApproachA',
                     '2': 'possibility_4_ApproachA',
                     }


# dir_path = 'combined_nn_results/refined_models/multiple_runs2'
dir_path = 'combined_nn_results/refined_models/approachA'

error_list = []
model_names = []
for key, model_name in final_test_models.items():
    model_dir = f'{dir_path}/{model_name}'
    model_names.append(model_name)
    model_mean_error = get_multiple_runs_error(model_dir)
    error_list.append(model_mean_error)

errors_df = pd.DataFrame(error_list, columns=['mean', 'std', 'min', 'max', 'median'], index=model_names)
errors_df.to_csv(f'{dir_path}/final_errors.csv')
