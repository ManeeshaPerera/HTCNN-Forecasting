import pandas as pd
from geots2img import ImageGenerator
import numpy as np
import constants

# I have run the cluster data creation notebook in  geo-timeseries-to-image project

# LON_RANGE = [114.73877508167377-.5, 121.47443279634425+.5]
# LAT_RANGE = [-34.800063682946856-.5, -28.87616188038324+0.5]


# # SWIS
LAT_RANGE = [-36.05, -26.5]
LON_RANGE = [113.8, 122.4]
GEO_RES = 0.005
UNIQUE_WEATHER_PCS = {0: 6028, 2: 6254, 3: 6430, 4: 6006, 10: 6324, 13: 6284, 14: 6173, 15: 6401, 16: 6426, 17: 6312,
                      18: 6230, 19: 6317, 20: 6112, 21: 6055,
                      24: 6509, 25: 6391, 26: 6528, 27: 6220, 28: 6280, 29: 6000}

WEATHER_MAP = {7: 'temperature', 8: 'humidity', 9: 'dewPoint', 10: 'wind', 11: 'pressure', 12: 'cloudCover',
               13: 'uvIndex'}
source_points = []
NUM_DAYS = 6
NUM_DAYS_WEATHER = 6



def get_weather_df(column_index, column_name):
    all_temp = []
    for cluster in UNIQUE_WEATHER_PCS:
        pc = UNIQUE_WEATHER_PCS[cluster]
        wc_data = pd.read_csv(f'swis_ts_data/ts_data/{pc}.csv', index_col=0).iloc[:,
                  column_index:column_index + 1]
        wc_data = wc_data.rename(columns={column_name: str(cluster)})
        all_temp.append(wc_data)
    return pd.concat(all_temp, axis=1)


def scale_data(df, num_days):
    scaled = df.divide(df.rolling((18 * num_days) + 1).max())
    scaled.dropna(inplace=True)
    return scaled


cluster_gen_data = pd.read_csv('swis_ts_data/geo_ts_cluster.csv', index_col=0)[18*6:]
coordinates_clusters = pd.read_csv('swis_ts_data/coordinates.csv')

# starting the forecasting day from 20th 2nd Feb
grid_data = pd.read_csv('swis_ts_data/ts_data/grid.csv', index_col=0).iloc[18:, 0:1]
test = grid_data[-18 * constants.TEST_DAYS:]
test_start_date = test.index[0]

# reading weather data
weather_data = []

print(grid_data)
print(cluster_gen_data)

for weather_index in range(7, 14):
    weather_data.append(get_weather_df(weather_index, WEATHER_MAP[weather_index]))

scaled_cluster_gen = scale_data(cluster_gen_data, NUM_DAYS)
scaled_grid_data = scale_data(grid_data, NUM_DAYS)
scaled_weather_data = []


# print(scaled_grid_data)
# print(scaled_cluster_gen)


for weather_dfs in weather_data:
    scaled_weather_data.append(scale_data(weather_dfs, NUM_DAYS_WEATHER))

# print(scaled_weather_data)


for cluster_num in UNIQUE_WEATHER_PCS:
    [lon, lat] = coordinates_clusters.loc[cluster_num, ['lon', 'lat']]
    source_points.append((lon, lat))


# we need an image generator for each time series
power_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)
temperature_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)
humidity_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)
dewPoint_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)
wind_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)
pressure_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)
cloudCover_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)
uvIndex_image = ImageGenerator(LON_RANGE, LAT_RANGE, GEO_RES)

power_image.set_source_points(source_points)
weather_generators = [temperature_image, humidity_image, dewPoint_image, wind_image, pressure_image,
                        cloudCover_image, uvIndex_image]

for image_gen in weather_generators:
    image_gen.set_source_points(source_points)

all_days = []
all_days_test = []
grid_label_train = []
grid_label_test = []
time = 0

while time < len(scaled_grid_data):
    grid_date = scaled_grid_data.index[time]
    all_features_per_day = []
    for time_per_day_weather in range(time, time+18):
        all_features = []
        for image_gen_index in range(0, len(weather_generators)):
            image_generator = weather_generators[image_gen_index]
            data_weather = scaled_weather_data[image_gen_index]
            time_point = data_weather.index[time_per_day_weather]
            image_generator.set_source_values(np.array(data_weather.loc[time_point].values))
            # generate values
            image_generator.generate_fitted_values()
            # get the fitted values
            vals = image_generator.get_fitted_values()
            # vals = np.array([[1,2], [5,6], [4,7]])
            h = len(vals)
            w = len(vals[0])
            all_features.append(vals.reshape(h, w, 1, 1))
        # power generator
        time_point_power = scaled_cluster_gen.index[time_per_day_weather]
        power_image.set_source_values(np.array(scaled_cluster_gen.loc[time_point_power].values))
        # generate values
        power_image.generate_fitted_values()
        # get the fitted values
        vals = power_image.get_fitted_values()
        # vals = np.array([[1,2], [5,6], [4,7]])
        h = len(vals)
        w = len(vals[0])
        all_features.append(vals.reshape(h, w, 1, 1))
        all_features_per_day.append(np.concatenate(all_features, axis=3))
    if grid_date < test_start_date:
        all_days.append(np.concatenate(all_features_per_day, axis=2))
        grid_label_train.append(scaled_grid_data[time: time+18].values)
        time = time + 1
    elif grid_date == test_start_date:
        all_days_test.append(np.concatenate(all_features_per_day, axis=2))
        grid_label_test.append(scaled_grid_data[time: time+18].values)
        time = time + 18
    else:
        all_days_test.append(np.concatenate(all_features_per_day, axis=2))
        grid_label_test.append(scaled_grid_data[time: time+18].values)
        time = time + 18


train_data = np.array(all_days, dtype=np.float32)
train_label = np.array(grid_label_train, dtype=np.float32)

test_data = np.array(all_days_test, dtype=np.float32)
test_label = np.array(grid_label_test, dtype=np.float32)


print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)


# with open('swis_ts_data/train_data.npy', 'wb') as f:
#     data_to_store = np.array(grid_label_test, dtype=np.float32)
#     np.save(f, data_to_store)
#     f.close()

with open('swis_ts_data/train_data.npy', 'wb') as train_file:
    np.save(train_file, train_data)
    train_file.close()

with open('swis_ts_data/train_label.npy', 'wb') as train_label_file:
    np.save(train_label_file, train_label)
    train_label_file.close()


with open('swis_ts_data/test_data.npy', 'wb') as test_data_file:
    np.save(test_data_file, test_data)
    test_data_file.close()


with open('swis_ts_data/test_label.npy', 'wb') as test_label_file:
    np.save(test_label_file, test_label)
    test_label_file.close()

