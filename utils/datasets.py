"""Dataset Class for OpenPack dataset.
"""
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openpack_toolkit as optk
import torch
import pandas as pd
import numpy as np
from omegaconf import DictConfig, open_dict
from openpack_toolkit import OPENPACK_OPERATIONS

logger = getLogger(__name__)


class OpenPackAll(torch.utils.data.Dataset):
    """Dataset class for IMU data.

    Attributes:
        data (List[Dict]): each sequence is stored in dict. The dict has 5 keys (i.e.,
            user, session, data, label(=class index), unixtime). data is a np.ndarray with
            shape = ``(N, channel(=acc_x, acc_y, ...), window, 1)``.
        index (Tuple[Dict]): sample index. A dict in this tuple as 3 property.
            ``seq`` = sequence index, ``sqg`` = segment index which is a sequential number
            within the single sequence. ``pos`` = sample index of the start of this segment.
        classes (optk.ActSet): list of activity classes.
        window (int): sliding window size.
        debug (bool): If True, enable debug mode. Default to False.
        submission (bool): Set True when you make submission file. Annotation data will not be
            loaded and dummy data will be generated. Default to False.

    Todo:
        * Make a minimum copy of cfg (DictConfig) before using in ``load_dataset()``.
        * Add method for parameter validation (i.e., assert).
    """
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
            self,
            cfg: DictConfig,
            user_session_list: Tuple[Tuple[int, int], ...],
            classes: optk.ActSet = OPENPACK_OPERATIONS,
            window: int = 30 * 60,
            window_keypoint: int = 15 * 60,
            submission: bool = False,
            debug: bool = False,
    ) -> None:
        """Initialize OpenPackImu dataset class.

        Args:
            cfg (DictConfig): instance of ``optk.configs.OpenPackConfig``. path, dataset, and
                annotation attributes must be initialized.
            user_session (Tuple[Tuple[int, int], ...]): the list of pairs of user ID and session ID
                to be included.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
            window (int, optional): window size [steps]. Defaults to 30*60 [s].
            submission (bool, optional): Set True when you want to load test data for submission.
                If True, the annotation data will no be replaced by dummy data. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
        """
        super().__init__()
        self.classes = classes
        self.window = window
        self.window_keypoint = window_keypoint
        self.submission = submission
        self.debug = debug

        self.load_dataset(
            cfg,
            user_session_list,
            window,
            submission=submission)

        self.preprocessing()

    def create_pd_from_data(self,data_array, time_array):
        values_df = pd.DataFrame(np.reshape(data_array, (len(data_array), -1)))
        time_df = pd.DataFrame({'unixtime': time_array}, index=values_df.index)
        df = pd.concat([values_df, time_df], axis=1)
        return df
    
    def merge_pds(self,df1, df2):
        merged_df = pd.merge_asof(df1, df2, on='unixtime')
        merged_df = merged_df.interpolate()
        merged_df = merged_df.ffill().bfill()
        return merged_df
    
    def load_dataset(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...],
        window: int = None,
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.

        Args:
            user_session (Tuple[Tuple[str, str], ...]): _description_
            window (int, optional): _description_. Defaults to None.
            submission (bool, optional): _description_. Defaults to False.
        """
        print("load dataset")
        data, index = [], []
        for seq_idx, (user, session) in enumerate(user_session_list):
            print(f"user {user}, session {session}")
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session

            path_keypoints = Path(
                    cfg.dataset.stream.path_keypoint.dir,
                    cfg.dataset.stream.path_keypoint.fname,
                )
            ts_sess_keypoints, x_sess_keypoints = optk.data.load_keypoints(path_keypoints)
            x_sess_keypoints = x_sess_keypoints[:(x_sess_keypoints.shape[0] - 1)]  # Remove prediction score.
            #print("Shape of x keypoints:", x_sess_keypoints.shape)
            #print("Shape of t keypoints:", ts_sess_keypoints.shape)
            
            x_sess_keypoints = x_sess_keypoints.transpose(2, 0, 1).reshape(34, -1).transpose(1,0)

            #print("Shape of x keypoints:", x_sess_keypoints.shape)

            paths_imu = []
            for device in cfg.dataset.stream.devices:
                with open_dict(cfg):
                    cfg.device = device

                path_imu = Path(
                    cfg.dataset.stream.path_imu.dir,
                    cfg.dataset.stream.path_imu.fname
                )
                paths_imu.append(path_imu)

            ts_sess_imu, x_sess_imu = optk.data.load_imu(
                paths_imu,
                use_acc=cfg.dataset.stream.acc,
                use_gyro=cfg.dataset.stream.gyro,
                use_quat=cfg.dataset.stream.quat)
            
            x_sess_imu = x_sess_imu.transpose(1,0)

            #print("Shape of x imu:", x_sess_imu.shape)
            #print("Shape of t imu:", ts_sess_imu.shape)

            #print("First t imu entries", ts_sess_imu)   
            #print("First t keypoints entries", ts_sess_keypoints)   


            imu_pd = self.create_pd_from_data(x_sess_imu, ts_sess_imu)
            keypoints_pd = self.create_pd_from_data(x_sess_keypoints, ts_sess_keypoints)
            merged_pd = self.merge_pds(imu_pd, keypoints_pd)
            print(merged_pd.shape)
            #print(merged_pd.head())

            assert merged_pd.shape[0] == ts_sess_imu.shape[0], "DataFrame and array are not of the same length"

            merged_pd = merged_pd.drop("unixtime", axis=1)
            path = Path(
                    cfg.dataset.annotation.path.dir,
                    cfg.dataset.annotation.path.fname
                )

            df_label_imu = optk.data.load_and_resample_operation_labels(
                    path, ts_sess_imu, classes=self.classes)
            
            #df_label_keypoints = optk.data.load_and_resample_operation_labels(path, ts_sess_keypoints, classes=self.classes)
            
            #print("IMU time", df_label_imu["annot_time"])
            #print("Keypoints time", df_label_keypoints["annot_time"])

            x_total = merged_pd.values.transpose(1,0)
            

            #print("Shape of the merged array:", x_total.shape)


            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sess_imu),), dtype=np.int64)
            else:
                path = Path(
                    cfg.dataset.annotation.path.dir,
                    cfg.dataset.annotation.path.fname
                )
                df_label_imu = optk.data.load_and_resample_operation_labels(
                    path, ts_sess_imu, classes=self.classes)
                label = df_label_imu["act_idx"].values

            data.append({
                "user": user,
                "session": session,
                "data": x_total,
                "label": label,
                "unixtime": ts_sess_imu,
            })

            seq_len = ts_sess_imu.shape[0]
            index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                      for seg_idx, pos in enumerate(range(0, seq_len, window))]
        self.data = data
        self.index = tuple(index)
        print("index:" )
        print(index)

    
    def load_dataset_keypoints(
        self,
        cfg: DictConfig,
        user_session: Tuple[Tuple[int, int], ...],
        submission: bool = False,
    ):
        data, index = [], []
        for seq_idx, (user, session) in enumerate(user_session):
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session

            path = Path(
                cfg.dataset.stream.path.dir,
                cfg.dataset.stream.path.fname,
            )
            ts_sess, x_sess = optk.data.load_keypoints(path)
            x_sess = x_sess[:(x_sess.shape[0] - 1)]  # Remove prediction score.

            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sess),), dtype=np.int64)
            else:
                path = Path(
                    cfg.dataset.annotation.path.dir,
                    cfg.dataset.annotation.path.fname
                )
                df_label = optk.data.load_and_resample_operation_labels(
                    path, ts_sess, classes=self.classes)
                label = df_label["act_idx"].values

            data.append({
                "user": user,
                "session": session,
                "data": x_sess,
                "label": label,
                "unixtime": ts_sess,
            })

            seq_len = x_sess.shape[1]
            index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                      for seg_idx, pos in enumerate(range(0, seq_len, self.window))]

        self.data = data
        self.index = tuple(index)

    def preprocessing(self) -> None:
        """This method is called after ``load_dataset()`` and apply preprocessing to loaded data.
        """
        logger.warning("No preprocessing is applied.")

    @property
    def num_classes(self) -> int:
        """Returns the number of classes

        Returns:
            int
        """
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackImu("
            f"index={len(self.index)}, "
            f"num_sequence={len(self.data)}, "
            f"submission={self.submission}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_dict = self.data[seq_idx]
        seq_len = seq_dict["data"].shape[1]

        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        print(f'Index:{index}, seq_idx: {seq_idx}, seg_idx: {seg_idx}, seq_len {seq_len}, head: {head}, tail: {tail}')
        if tail >= seq_len:
            print("inside")
            pad_tail = tail - seq_len
            tail = seq_len
        else:
            pad_tail = 0
        assert (
            head >= 0) and (
            tail > head) and (
            tail <= seq_len), f"head={head}, tail={tail}"

        x = seq_dict["data"][:, head:tail, np.newaxis]
        t = seq_dict["label"][head:tail]
        ts = seq_dict["unixtime"][head:tail]

        if pad_tail > 0:
            x = np.pad(x, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            t = np.pad(t, [(0, pad_tail)], mode="constant",
                       constant_values=self.classes.get_ignore_class_index())
            ts = np.pad(ts, [(0, pad_tail)],
                        mode="constant", constant_values=ts[-1])

        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        ts = torch.from_numpy(ts)
        return {"x": x, "t": t, "ts": ts}





    

# -----------------------------------------------------------------------------


