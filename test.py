import torch
from pathlib import Path

import numpy as np
import pandas as pd
import json
import openpack_toolkit as optk
from omegaconf import DictConfig, OmegaConf, open_dict
from openpack_toolkit.utils.notebook import noglobal

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
sns.set("notebook", "whitegrid", font_scale=1.5)


x = torch.rand(5, 3)
print(x)

DATASET_ROOTDIR = "C:\\Users\\Ego\\Documents\\TFM\\dataset"
OPENPACK_VERSION = "v0.3.1"

cfg = OmegaConf.create({
    "user": optk.configs.users.U0102,
    "session": None,
    "path": {
        "openpack": {
            "version": OPENPACK_VERSION,
            "rootdir": DATASET_ROOTDIR + "/openpack/${.version}",
        },
    },
    "dataset": {
        "annotation": None,
        "stream": None,
    }
})

print(OmegaConf.to_yaml(cfg))
print(OmegaConf.to_yaml(optk.OPENPACK_OPERATIONS))
print(torch.cuda.is_available())