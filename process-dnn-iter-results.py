import pandas as pd
import constants as const

if __name__ == '__main__':
    filepath = 'dnn_results/lstm/'
    for ts in const.TS:
        df_list = []
        for iter_num in range(1, 6):
            file = pd.read_csv(f'{filepath}{ts}/forecasts_iteration_{iter_num}.csv', index_col=[0])
            file.columns = [f'fc_{iter_num}']
            df_list.append(file)

        df = pd.concat(df_list, axis=1)
        df['average_fc'] = df.median(axis=1)
        df.to_csv(f'{filepath}final_results/{ts}.csv')

