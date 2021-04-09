import constants as const
import pandas as pd

if __name__ == '__main__':
    for ts in const.TS:
        filepath = f'benchmark_results/tbats'
        results = pd.read_csv(f'{filepath}/final_results/{ts}.csv', index_col=[0])
        # save the old file
        results.to_csv(f'{filepath}/{ts}.csv')
        # remove negative values
        results['average_fc'][results['average_fc'] < 0] = 0
        # save the new file
        results.to_csv(f'{filepath}/final_results/{ts}.csv')
