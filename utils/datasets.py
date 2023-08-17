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
from .dataloader import load_e4acc

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
            #print(merged_pd.shape)
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
        print (len(self.index))
        print("data:" )
        print(len(data))
        #print(index)

    
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
        print(f"getitem {index}")
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_dict = self.data[seq_idx]
        seq_len = seq_dict["data"].shape[1]

        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        #print(f'Index:{index}, seq_idx: {seq_idx}, seg_idx: {seg_idx}, seq_len {seq_len}, head: {head}, tail: {tail}')
        if tail >= seq_len:
            #print("inside")
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

        print(f"x t shapes {x.shape} {t.shape}")
        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        ts = torch.from_numpy(ts)
        
        return {"x": x, "t": t, "ts": ts}


class OpenPackAllSplit(torch.utils.data.Dataset):    
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
            self,
            cfg: DictConfig,
            user_session_list: Tuple[Tuple[int, int], ...],
            classes: optk.ActSet = OPENPACK_OPERATIONS,
            window_imu: int = 30 * 60,
            window_keypoint: int = 15 * 60,
            window_e4: int = 32 * 60,
            submission: bool = False,
            debug: bool = False,
    ) -> None:       
        super().__init__()
        self.classes = classes
        self.window_imu = window_imu
        self.window_keypoint = window_keypoint
        self.window_e4 = window_e4
        self.submission = submission
        self.debug = debug

        self.load_dataset(
            cfg,
            user_session_list,
            window_imu,
            window_keypoint,
            window_e4,
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
        window_imu: int = None,
        window_keypoint: int = None,
        window_e4: int = None,
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.

        Args:
            user_session (Tuple[Tuple[str, str], ...]): _description_
            window (int, optional): _description_. Defaults to None.
            submission (bool, optional): _description_. Defaults to False.
        """
        print("load dataset")
        #data, index = [], []
        data = dict(imu=[], keypoints=[], e4=[])
        labels = dict(imu=[], keypoints=[], e4=[])
        index = 0
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
            
            x_sess_keypoints = x_sess_keypoints.transpose(2, 0, 1).reshape(34, -1)

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

            paths_e4 = []
            for device in cfg.dataset.stream.devices_e4:
                with open_dict(cfg):
                    cfg.device=device

                path_e4 = Path(
                    cfg.dataset.stream.path_e4.dir,
                    cfg.dataset.stream.path_e4.fname
                )
                paths_e4.append(path_e4)
            
            ts_sess_e4, x_sess_e4 = load_e4acc(paths_e4)

            path = Path(
                    cfg.dataset.annotation.path.dir,
                    cfg.dataset.annotation.path.fname
                )

            df_label_imu = optk.data.load_and_resample_operation_labels(
                    path, ts_sess_imu, classes=self.classes)
            labels_imu = df_label_imu["act_idx"].values
            
            df_label_keypoints = optk.data.load_and_resample_operation_labels(path, ts_sess_keypoints, classes=self.classes)
            labels_keypoints = df_label_keypoints["act_idx"].values

            df_label_e4 = optk.data.load_and_resample_operation_labels(path, ts_sess_e4, classes=self.classes)
            labels_e4 = df_label_e4["act_idx"].values

            index_imu, index_kp, index_e4 = 0,0,0
            remain_imu, remain_kp, remain_e4 = True, True, True
            while (remain_imu or remain_kp or remain_e4):
                imu_dim, _ = x_sess_imu.shape
                kp_dim, _ = x_sess_keypoints.shape
                e4_dim, _ = x_sess_e4.shape

                if (index_imu + window_imu <= len(x_sess_imu)):
                    data["imu"][index] = x_sess_imu[:, index_imu:index_imu + window_imu]
                    labels["imu"][index] = labels_imu[index_imu:index_imu + window_imu]
                    index_imu = index_imu + window_imu
                else:
                    remain_imu = False
                    diff_imu = len(x_sess_imu) - index_imu
                    if (diff_imu > 0):
                        arr = x_sess_imu[:, index_imu:-1]
                        padded_x = np.pad(arr, [(0, 0), (0, window_imu - diff_imu)], mode='constant', constant_values=0)
                        arr = labels_imu[index_imu:-1]
                        padded_label = np.pad(arr, (0, window_imu - diff_imu), mode='constant', constant_values=self.classes.get_ignore_class_index())
                        data["imu"][index] = padded_x
                        labels["imu"][index] = padded_label
                        index_imu = index_imu + window_imu
                    else:
                        data["imu"][index] = np.full((imu_dim, window_imu), 0)
                        labels["imu"][index] = np.full((1, window_imu), self.classes.get_ignore_class_index())

                if (index_kp + window_keypoint <= len(x_sess_keypoints)):
                    data["keypoints"][index] = x_sess_keypoints[:, index_kp:index_kp + window_keypoint]
                    labels["keypoints"][index] = labels_keypoints[index_kp:index_kp + window_keypoint]
                    index_kp = index_kp + window_keypoint
                else:
                    remain_kp = False
                    diff_kp = len(x_sess_keypoints) - index_kp
                    if (diff_kp > 0):
                        arr = x_sess_keypoints[:, index_kp:-1]
                        padded_x = np.pad(arr, [(0, 0), (0, window_keypoint - diff_kp)], mode='constant', constant_values=0)
                        arr = labels_keypoints[index_kp:-1]
                        padded_label = np.pad(arr, (0, window_keypoint - diff_kp), mode='constant', constant_values=self.classes.get_ignore_class_index())
                        data["keypoints"][index] = padded_x
                        labels["keypoints"][index] = padded_label
                        index_kp = index_kp + window_keypoint
                    else:
                        data["keypoints"][index] = np.full((kp_dim, window_keypoint), 0)
                        labels["keypoints"][index] = np.full((1, window_keypoint), self.classes.get_ignore_class_index())

                
                if (index_e4 + window_e4 <= len(x_sess_e4)):
                    data["e4"][index] = x_sess_e4[:, index_e4:index_e4 + window_e4]
                    labels["e4"][index] = labels_e4[index_e4:index_e4 + window_e4]
                    index_e4 = index_e4 + window_e4
                else:
                    remain_e4 = False
                    diff_e4 = len(x_sess_e4) - index_e4
                    if (diff_e4 > 0):
                        arr = x_sess_e4[:, index_e4:-1]
                        padded_x = np.pad(arr, [(0, 0), (0, window_e4 - diff_e4)], mode='constant', constant_values=0)
                        arr = labels_e4[index_e4:-1]
                        padded_label = np.pad(arr, (0, window_e4 - diff_e4), mode='constant', constant_values=self.classes.get_ignore_class_index())
                        data["e4"][index] = padded_x
                        labels["e4"][index] = padded_label
                        index_e4 = index_e4 + window_e4
                    else:
                        data["e4"][index] = np.full((e4_dim, window_e4), 0)
                        labels["e4"][index] = np.full((1, window_e4), self.classes.get_ignore_class_index())
                
                index += 1
            
        self.data = data
        self.labels = labels 
        self.length = index 
    
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
            f"index={self.length}, "
            f"num_sequence={len(self.data)}, "
            f"submission={self.submission}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        print(f"getitem {index}")

        x_keypoints = self.data["keypoints"][index]
        label_keypoints = self.labels["keypoints"][index]
        
        x_imu = self.data["imu"][index]
        label_imu = self.labels["imu"][index]

        x_e4 = self.data["e4"][index]
        label_e4 = self.labels["e4"][index]

        #print(f"x t shapes {x.shape} {t.shape}")
        x_keypoints = torch.from_numpy(x_keypoints)
        label_keypoints = torch.from_numpy(label_keypoints)

        x_imu =  torch.from_numpy(x_imu)
        label_imu =  torch.from_numpy(label_imu)

        x_e4 = torch.from_numpy(x_e4)
        label_e4 = torch.from_numpy(label_e4)
       
        
        return {"x_keypoints": x_keypoints, "label_keypoints": label_keypoints, "x_imu": x_imu, "label_imu": label_imu, "x_e4": x_e4, "label_e4": label_e4}


    

# -----------------------------------------------------------------------------


