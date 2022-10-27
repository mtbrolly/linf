"""
Load drifter data from file provided by AOML at NOAA per request made via:
    https://www.aoml.noaa.gov/phod/gdp/interpolated/data/subset.php .
The request made  on 12/04/2022 for all available _drogued_ drifter data.

Received the same day in the following email:
"
To download the data files(s) proceed as follows:

   By clicking on the following hyper-link(s)

https://www.aoml.noaa.gov/ftp/pub/od/envids/metadata_gld.20220412_094949.zip
https://www.aoml.noaa.gov/ftp/pub/od/envids/interpolated_gld.20220412_094949.zip
Or

   By using the following ftp instructions

      1. ftp.aoml.noaa.gov
      2. enter 'anonymous' for userid.
      3. enter your 'email address' for password.
      4. enter 'binary' to set the transfer type
      5. enter 'cd /od/pub/envids'
      6. enter 'get metadata_gld.20220412_094949.zip'
      6. enter 'get interpolated_gld.20220412_094949.zip'
      7. enter 'quit' to log off.

NOTICE:  files are removed 5 days after creation date.

Your submitted request was: request_gld.20220412_094949
Drogue=1
fromDate=1979/02/15
toDate=2021/09/30
northernEdge=90
southernEdge=-78
westernEdge=-180
easternEdge=180


For questions or comments, please contact:
Jay.Harris@noaa.gov
"

The links in the email above will die but the request detailed above can be
easily repeated. The main zipped file is approximately 500MB in size and
contains data from 24474 drifters.
"""

import pandas as pd
import numpy as np


def load_drifter_csv_as_df():
    fname = "data/GDP/raw/interpolated_gld.20220412_094949"
    return pd.read_csv(fname, sep='\s+', parse_dates=[[1, 2]])  # noqa: W605


def remove_drifters_with_gaps():
    df = load_drifter_csv_as_df()
    ids = np.unique(df.id[:])
    ids_gaps = []
    # gaps = np.array([], dtype=np.timedelta64)

    for i, idx in enumerate(ids):
        if np.mod(i, 1000) == 0:
            print(f"i = {i}")
        ts = df[df['id'] == idx]['date_time']
        sixhours = np.timedelta64(6, 'h')
        dts = ts[1:].values - ts[:-1].values
        if not all(dts == sixhours):
            ids_gaps.append(idx)
            # gaps = np.concatenate((gaps, (dts - sixhours)[dts > sixhours]))

    df_nogaps = df[~df['id'].isin(ids_gaps)]
    df_nogaps.to_pickle("data/GDP/drifters_without_gaps.pkl")


def process_raw_df(drop_extra_columns=True, drop_drifters_with_gaps=True):
    if drop_drifters_with_gaps:
        df = pd.read_pickle(
            "data/GDP/drifters_without_gaps.pkl")
    else:
        df = load_drifter_csv_as_df()

    if drop_extra_columns:
        df.drop(columns=['t', 've', 'vn', 'speed', 'varlat', 'varlon', 'vart'],
                inplace=True)

    df.set_index(['id', 'date_time'], inplace=True)
    df.sort_index(inplace=True)
    return df


def train_test_split_drifters():
    """
    Split dataframe in half by drifters.
    """
    df = process_raw_df()
    ids = np.unique(df.index.get_level_values(level=0))
    rng = np.random.default_rng(seed=1)
    train_ids = np.sort(rng.choice(ids, size=len(ids)//2, replace=False))
    test_ids = np.sort(np.setdiff1d(ids, train_ids))
    train_df = df[df.index.get_level_values(level=0).isin(train_ids)].copy()
    test_df = df[df.index.get_level_values(level=0).isin(test_ids)].copy()
    return train_df, test_df


def get_drifter_displacements_as_numpy(tau=56, overlapping=True):
    """
    Subsamples (X0, DX) pairs from buoy trajectories for intervals of length
    tau (in days).
    """
    data_dir = f"data/GDP/{tau:.0f}day/"
    sets = ["train", "test"]
    dfs = train_test_split_drifters()
    for i in range(2):
        df = dfs[i]
        dt = 0.25
        tau_dt = int(np.ceil(tau / dt))
        ids = np.unique([a[0] for a in df.index.values])

        for d in range(ids.size):
            if np.mod(d, 100) == 0:
                print(d)
            if overlapping:
                X0 = df.xs(ids[d], level=0)[:-tau_dt].to_numpy()
                DX = df.xs(ids[d], level=0)[tau_dt:].to_numpy() - X0
            else:
                sub = df.xs(ids[d], level=0)[::tau_dt]
                X0 = sub.values[:-1]
                DX = sub.values[1:] - X0
            if d == 0:
                X0agg = X0
                DXagg = DX
            else:
                X0agg = np.concatenate((X0agg, X0))
                DXagg = np.concatenate((DXagg, DX))

        del df
        np.save(data_dir + "X0raw_" + sets[i] + ".npy", X0agg)
        np.save(data_dir + "DXraw_" + sets[i] + ".npy", DXagg)


def reordering_and_dateline_wrap():
    data_dir = "data/GDP/56day/"

    X = np.load(data_dir + "X0raw_train.npy")
    XVAL = np.load(data_dir + "X0raw_test.npy")
    Y = np.load(data_dir + "DXraw_train.npy")
    YVAL = np.load(data_dir + "DXraw_test.npy")

    # Reorder data from (lat, lon) to (lon, lat) and deal with displacements
    # which cross the dateline.
    Xsets = [X, XVAL]
    Ysets = [Y, YVAL]

    for i, x in enumerate(Xsets):
        temp = x[:, 1].copy()
        x[:, 1] = x[:, 0]
        x[:, 0] = temp
        del temp, x

    for i, y in enumerate(Ysets):
        temp = y[:, 1].copy()
        y[:, 1] = y[:, 0]
        y[:, 0] = temp
        del temp
        y[:, 0] += (y[:, 0] < -270.) * 360. + (y[:, 0] > 270.) * (-360.)

    np.save(data_dir + "X0_train.npy", X)
    np.save(data_dir + "X0_test.npy", XVAL)
    np.save(data_dir + "DX_train.npy", Y)
    np.save(data_dir + "DX_test.npy", YVAL)
