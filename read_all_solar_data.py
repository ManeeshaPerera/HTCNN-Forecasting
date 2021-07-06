import pandas as pd
import os

DATA_PATH = '../_data/sites/'

# site1 = pd.read_csv(f'{DATA_PATH}/6041/1948717897-solar.csv')
# # site2 = pd.read_csv(f'{DATA_PATH}/6004/863982947-solar.csv')
# #
# print(len(site1))
# duplicateRowsDF = site1[site1.duplicated()]
# print(len(duplicateRowsDF))
# # print(site1.loc[site1['timestamp'] == '2020-04-05 02:00:00+08:00'])
#
# site1_no_dup = site1.drop_duplicates(subset=None, keep='first', inplace=False)
# print(len(site1_no_dup))
# duplicateRowsDF = site1_no_dup[site1_no_dup.duplicated()]
# print(len(duplicateRowsDF))
# print(site2)
#
#
# energy_site = site1.iloc[:, [0, 1]].rename(columns={'energy': '648402560'})
# power_site = site1.iloc[:, [0, 2]].rename(columns={'power': '648402560'})
#
# print(energy_site)
# print(power_site)
#
# energy_site2 = site2.iloc[:, [0, 1]].rename(columns={'energy': '863982947'})
# power_site2 = site2.iloc[:, [0, 2]].rename(columns={'power': '863982947'})
#
# df = energy_site.merge(energy_site2, how= 'outer')
# df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
# print(df)

all_data = []
site_ids = []
for subdir, dirs, files in os.walk(DATA_PATH):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".csv"):
            site_id = filename.split("-")[0]
            if site_id not in site_ids:
                site_ids.append(site_id)
                all_data.append(filepath)

print("all site: ", len(all_data))
print(len(site_ids))

all_dfs_energy = []
all_dfs_power = []

site_index = 0
for file_path_site in all_data:
    data_site = pd.read_csv(file_path_site)
    data_site = data_site.drop_duplicates(subset='timestamp', keep='first', inplace=False)

    energy_site = data_site.iloc[:, [0, 1]]
    power_site = data_site.iloc[:, [0, 2]]

    energy_site['site_id'] = site_ids[site_index]
    power_site['site_id'] = site_ids[site_index]

    all_dfs_energy.append(energy_site)
    all_dfs_power.append(power_site)
    site_index += 1

all_energy = pd.concat(all_dfs_energy)
print(all_energy)

# all_energy2 = all_energy[all_energy.duplicated(subset=['timestamp', 'site_id'], keep=False)]
# print(all_energy2['site_id'].unique())
# print(all_energy2)

all_energy = all_energy.pivot(index='timestamp', columns='site_id', values='energy')
# print(all_energy)
all_energy.index = [pd.Timestamp(x) for x in all_energy.index]
all_energy.index = all_energy.index.rename('timestamp')
print(all_energy)

all_energy.to_csv('all_ts_data/energy.csv')
all_energy.to_pickle('all_ts_data/all_data_energy')

all_power = pd.concat(all_dfs_power)
all_power = all_power.pivot(index='timestamp', columns='site_id', values='power')
print(all_power)
all_power.index = [pd.Timestamp(x) for x in all_power.index]
all_power.index = all_power.index.rename('timestamp')
print(all_power)

all_power.to_csv('all_ts_data/power.csv')
all_power.to_pickle('all_ts_data/all_power_energy')
