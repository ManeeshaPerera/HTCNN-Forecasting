from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd


def plot_hf(data, pc_order, pc_map):
    figure_hf = make_subplots(rows=5, cols=6, shared_xaxes=True,
                              specs=[[{"colspan": 6}, None, None, None, None, None],
                                     [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
                                     [{"colspan": 2}, None, {}, {"colspan": 2}, None,
                                      {}], [{}, {}, {}, {}, {}, {}],
                                     [{}, {}, {}, {}, {}, {}]],
                              subplot_titles=(
                                  "Grid", "Transmission Line 1 (132 kv)", "Transmission Line 2 (66 kv)", "Substation 1",
                                  "Substation 2", "Substation 3", "Substation 4",
                                  "Postcode 6010", "Postcode 6014", "Postcode 6011", "Postcode 6280",
                                  "Postcode 6281", "Postcode 6284"))

    df_col = 1
    row = 1
    col = 1
    for hf_val in data.columns[0:13]:
        # print(df_col)
        # print(row, col)
        figure_hf.add_trace(go.Scatter(
            x=data[hf_val].index,
            y=data[hf_val].values,
            name=hf_val,
            showlegend=False
        ), row=row, col=col)
        if df_col == 1 or df_col == 3 or df_col == 7 or df_col == 13:
            row = row + 1
            col = 1
        elif df_col == 2:
            col = col + 3
        elif df_col == 4 or df_col == 6:
            col = col + 2
        else:
            col = col + 1
        df_col = df_col + 1

    row_site = 5
    col_site = 1

    for pc in pc_order:
        sites = pc_map[pc]
        for site in sites:
            site_data = data[site]
            figure_hf.add_trace(go.Scatter(
                x=site_data.index,
                y=site_data.values,
                showlegend=False
            ), row=row_site, col=col_site)
        col_site = col_site + 1
    figure_hf.write_html('../ts_data/hf_rnn_new.html')
    # figure_hf.write_image("../ts_data/hf.png")


if __name__ == '__main__':
    data = pd.read_pickle('../input/ts_1h')
    pc_order = data.columns[7:7 + 6].tolist()
    site_info = pd.read_csv('../input/site_info.csv')
    sites = data.columns[7 + 6:]
    pc_map = {}

    for site in sites:
        pc = site_info.loc[site_info['label'] == int(site)]['postcode'].values[0]
        pcs_checked = pc_map.keys()
        if pc not in pcs_checked:
            pc_map[pc] = [site]
        else:
            pc_map[pc].append(site)
    print(pc_map)
    for val in pc_map:
        print(len(pc_map[val]))
    print(pc_order)
    plot_hf(data, pc_order, pc_map)
