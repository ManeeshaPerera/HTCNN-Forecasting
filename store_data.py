import pandas as pd


def shift_forecast_days(data, horizon):
    dfs = [data]
    for i in range(1, horizon + 1):
        df1 = data.shift(-i)
        df1.columns = [col + "_" + str(i) for col in data.columns]
        dfs.append(df1.iloc[:,1:])
    new_data = pd.concat(dfs, axis=1).dropna()
    return new_data


if __name__ == '__main__':
    # read the solar time series dataframe
    h_ts = pd.read_pickle('input/ts_1h')
    site_info = pd.read_csv('input/site_info.csv')
    horizon = 14
    print(len(h_ts))
    print(len(h_ts.columns))
    number_of_pcs = 6
    site_ids = h_ts.columns[7 + number_of_pcs:]
    print("number of sites: ", len(site_ids))
    checked_sites = []

    for col in h_ts.columns[:7]:
        # chunck the data as the time series with weather will not have data for the forecast days
        power_data = h_ts[[col]][0:-horizon]
        power_data.columns = ['power']
        power_data.to_pickle(f'ts_data/{col}')
        power_data.to_csv(f'ts_data/{col}.csv')
        print(len(power_data))

    for col in h_ts.columns[7:7 + number_of_pcs]:
        power_data = h_ts[[col]]
        power_data.columns = ['power']
        weather_data = pd.read_pickle(f'input/weather_{col}')
        weather_data = weather_data.reset_index()
        weather_data['date_str'] = [x.tz_localize(None) for x in weather_data['date']]
        weather_data = weather_data.set_index('date_str')
        weather_data = weather_data.drop(columns=['date', 'timestamp'])
        pc_data = pd.concat([power_data, weather_data], axis=1).dropna()
        pc_data = shift_forecast_days(pc_data, horizon)
        print(len(pc_data))
        pc_data.to_pickle(f'ts_data/{col}')
        pc_data.to_csv(f'ts_data/{col}.csv')

        for site in site_ids:
            if site not in checked_sites:
                pc = site_info.loc[site_info['label'] == int(site)]['postcode'].values[0]
                if int(pc) == int(col):
                    power_site = h_ts[[site]]
                    power_site.columns = ['power']
                    site_data = pd.concat([power_site, weather_data], axis=1).dropna()
                    site_data = shift_forecast_days(site_data, horizon)
                    print(len(site_data))
                    checked_sites.append(site)
                    site_data.to_pickle(f'ts_data/{site}')
                    site_data.to_csv(f'ts_data/{site}.csv')
