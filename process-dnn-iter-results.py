import pandas as pd
import constants as const
import src.utils as utils

if __name__ == '__main__':
    filepath = 'dnn_results/lstm/'
    for ts in const.TS[0:1]:
        data = pd.read_csv(f'ts_data/{ts}.csv', index_col=[0])
        train, test = utils.split_hourly_data_for_stat_models(data)

        df_list = [test]
        for iter_num in range(1, 4):
            file = pd.read_csv(f'{filepath}{ts}/forecasts_iteration_{iter_num}.csv', index_col=[0])
            file.columns = [f'fc_{iter_num}']
            df_list.append(file)

        df = pd.concat(df_list, axis=1)
        df['average_fc'] = df.iloc[:, 1:].median(axis=1)
        df.to_csv(f'{filepath}final_results/{ts}.csv')

