import copy
import math
import multiprocessing
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from utils.datasets import OpenPackAll, OpenPackAllSplit, OpenPackSensorFusion, OpenPackSensorFusionWithKeypoints


class OpenPackAllDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = OpenPackAll

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        kwargs = {
            "window": self.cfg.train.window,
            "debug": self.cfg.debug,
        }
        return kwargs
    
class OpenPackAllSplitDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = OpenPackAllSplit

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        kwargs = {
            "debug": self.cfg.debug
        }
        return kwargs
    
class OpenPackSensorFusionDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = OpenPackSensorFusion

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        kwargs = {
            "window": self.cfg.train.window,
            "debug": self.cfg.debug
        }
        return kwargs
    

class OpenPackSensorFusionWithKeypointsDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = OpenPackSensorFusionWithKeypoints

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        kwargs = {
            "window": self.cfg.train.window,
            "debug": self.cfg.debug
        }
        return kwargs