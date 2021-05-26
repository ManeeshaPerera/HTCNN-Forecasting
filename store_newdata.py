import pandas as pd
import constants as const


def shift_solar_data(power_data):
    # day 1 will be the power value as the window generator chuncks that

    day2 = power_data.shift(14)
    day2.columns = ['day2']

    day3 = day2.shift(14)
    day3.columns = ['day3']

    day4 = day3.shift(14)
    day4.columns = ['day4']

    day5 = day4.shift(14)
    day5.columns = ['day5']

    day6 = day5.shift(14)
    day6.columns = ['day6']

    day7 = day6.shift(14)
    day7.columns = ['day7']

    fc_data = pd.concat([power_data, day2, day3, day4, day5, day6, day7], axis=1).dropna()
    return fc_data


def process_weather(weather_original):
    weather_original = weather_original.reset_index()
    weather_original['date_str'] = [x.tz_localize(None) for x in weather_original['date']]
    weather_original = weather_original.set_index('date_str')
    weather_original = weather_original.drop(columns=['date', 'timestamp'])
    return weather_original


def get_key(site_id):
    for pc, site_list in const.SITE_MAP.items():
        for site in site_list:
            if site_id == site:
                return pc


if __name__ == '__main__':
    # read the solar time series dataframe
    index = 0
    for ts in const.TS:
        print(ts)
        data = pd.read_pickle(f'ts_data/{ts}')
        if index <= 6:
            # we are only using power data
            new_data = shift_solar_data(data)
            new_data.to_csv(f'ts_data/new/{ts}.csv')
            new_data.to_pickle(f'ts_data/new/{ts}')

        else:
            # starting from postcode level
            power_df = data.iloc[:, 0:1]
            new_data = shift_solar_data(power_df)
            if index > 12:
                pc = get_key(ts)
            else:
                pc = ts
            weather_all = pd.read_pickle(f'input/weather_{pc}')
            weather_all = process_weather(weather_all)
            # shift to get the weather of the forecasting day
            weather = data.iloc[:, 1:8].shift(-14)
            weather = weather.dropna()

            # now I get data for 02-13 which will actually be data of 02-14 because that's the forecasting day
            weather_day = weather_all['2021-02-13 00:00:00': '2021-02-14 23:00:00'].shift(-24).dropna()[
                          '2021-02-13 07:00:00': '2021-02-13 20:00:00']
            weather = weather.append(weather_day)
            new_ts_data = pd.concat([new_data, weather['2020-02-19 07:00:00':'2021-02-13 20:00:00']], axis=1)
            new_ts_data.to_csv(f'ts_data/new/{ts}.csv')
            new_ts_data.to_pickle(f'ts_data/new/{ts}')

        index = index + 1
