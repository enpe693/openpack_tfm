from typing import List, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

def load_e4acc(
    paths: Union[Tuple[Path, ...], List[Path]],
    th: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load e4acc data from CSVs.

    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(
        paths, (tuple, list)
    ), f"the first argument `paths` expects tuple of Path, not {type(paths)}."

    channels = ["acc_x", "acc_y", "acc_z"]

    ts_ret, x_ret, ts_list = None, [], []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            return None, None
        assert set(channels) < set(df.columns)

        ts = df["time"].values
        x = df[channels].values.T

        ts_list.append(ts)
        x_ret.append(x)

    min_len = min([len(ts) for ts in ts_list])
    ts_ret = None
    for i in range(len(paths)):
        x_ret[i] = x_ret[i][:, :min_len]
        ts_list[i] = ts_list[i][:min_len]

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], " f"but difference smaller than th={th} is allowed."
            )

    x_ret = np.concatenate(x_ret, axis=0)
    return ts_ret, x_ret