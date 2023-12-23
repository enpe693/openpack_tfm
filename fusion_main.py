from logging import getLogger
from pathlib import Path
from typing import Dict, Optional
from utils.datamodule import *
from model.lightning_module import *

import hydra
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    construct_submission_dict, eval_operation_segmentation_wrapper,
    make_submission_zipfile)

logger = getLogger(__name__)
tensorboard_logger = TensorBoardLogger("tb_logs", name="fusion_competition_4")
optorch.configs.register_configs()
optorch.utils.reset_seed()


def save_training_results(log: Dict, logdir: Path) -> None:
    """
    Created from the code in https://github.com/open-pack/openpack-torch/tree/main/examples
    """
    # -- Save Model Outputs --
    df = pd.concat(
        [
            pd.DataFrame(log["train"]),
            pd.DataFrame(log["val"]),
        ],
        axis=1,
    )
    df.index.name = "epoch"

    path = Path(logdir, "training_log.csv")
    df.to_csv(path, index=True)
    logger.debug(f"Save training logs to {path}")
    print(df)


def train(cfg: DictConfig):
    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)
    logger.debug(f"logdir = {logdir}")
    optk.utils.io.cleanup_dir(logdir, exclude="hydra")


    #Change to your local directory
    imu_path = "...\\MyDeepConvLSTMLM\\MyDeepConvLSTMLM_imu_all\\tb_logs\\imu_all\\version_0\\checkpoints"
    e4_path = "...\\MyDeepConvLstm\\train_e4_100\\tb_logs\\e4_deepconvlstm_100\\version_0\\checkpoints"
    keypoints_path = "...\\MyDeepConvLstm\\keypoints\\tb_logs\\keypoints_100\\version_0\\checkpoints"

    e4_path = Path(e4_path, "last.ckpt")
    imu_path = Path(imu_path, "last.ckpt")
    keypoints_path = Path(keypoints_path, "last.ckpt")

    cfg.model_type ="e4"
    e4_model = MyDeepConvLSTMLM.load_from_checkpoint(e4_path, cfg=cfg)
    cfg.model_type ="imu"
    imu_model = MyDeepConvLSTMLM.load_from_checkpoint(imu_path, cfg=cfg)
    cfg.model_type ="keypoints"
    keypoints_model = MyDeepConvLSTMLM.load_from_checkpoint(keypoints_path, cfg=cfg)

    #datamodule = OpenPackAllDataModule(cfg)
    datamodule = OpenPackAllSplitDataModule(cfg)
    plmodel = FusionOfModelsLM(keypoints_model=keypoints_model, imu_model=imu_model, e4_model=e4_model,cfg=cfg)
    #plmodel = SplitDataModelLM(cfg)
    plmodel.to(dtype=torch.float, device=device)
    logger.info(plmodel)

    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        monitor=None,
    )

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=num_epoch,
        logger=tensorboard_logger,  
        default_root_dir=logdir,
        enable_progress_bar=False,  
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    print("training")
    logger.debug(f"logdir = {logdir}")

    logger.info(f"Start training for {num_epoch} epochs.")
    trainer.fit(plmodel, datamodule)
    logger.info("Finish training!")

    logger.debug(f"logdir = {logdir}")
    save_training_results(plmodel.log_dict, logdir)
    logger.debug(f"logdir = {logdir}")

def test(cfg: DictConfig, mode: str = "test"):
     """
    Created from the code in https://github.com/open-pack/openpack-torch/tree/main/examples
    """
    assert mode in ("test", "submission", "test-on-submission")
    logger.debug(f"test() function is called with mode={mode}.")

    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)

    #datamodule = OpenPackAllDataModule(cfg)
    datamodule = OpenPackAllSplitDataModule(cfg)
    datamodule.setup(mode)

    if (not cfg.model_path):
        ckpt_path = Path(logdir, "checkpoints", "last.ckpt")
    else:
        ckpt_path = Path(cfg.model_path, "last.ckpt")
    logger.info(f"load checkpoint from {ckpt_path}")
    
    #Change to your local directory
    imu_path = "...\\MyDeepConvLSTMLM\\MyDeepConvLSTMLM_imu_all\\tb_logs\\imu_all\\version_0\\checkpoints"
    e4_path = "...\\MyDeepConvLstm\\train_e4_100\\tb_logs\\e4_deepconvlstm_100\\version_0\\checkpoints"
    keypoints_path = "...\\MyDeepConvLstm\\keypoints\\tb_logs\\keypoints_100\\version_0\\checkpoints"

    e4_path = Path(e4_path, "last.ckpt")
    imu_path = Path(imu_path, "last.ckpt")
    keypoints_path = Path(keypoints_path, "last.ckpt")

    cfg.model_type ="e4"
    e4_model = MyDeepConvLSTMLM.load_from_checkpoint(e4_path, cfg=cfg)
    cfg.model_type ="imu"
    imu_model = MyDeepConvLSTMLM.load_from_checkpoint(imu_path, cfg=cfg)
    cfg.model_type ="keypoints"
    keypoints_model = MyDeepConvLSTMLM.load_from_checkpoint(keypoints_path, cfg=cfg)

    plmodel = FusionOfModelsLM.load_from_checkpoint(ckpt_path, cfg=cfg, keypoints_model=keypoints_model, imu_model=imu_model, e4_model=e4_model )
    plmodel.to(dtype=torch.float, device=device)

    trainer = pl.Trainer(
        gpus=[0],
        logger=False,  
        default_root_dir=None,
        enable_progress_bar=False,  
        enable_checkpointing=False,  
    )

    if mode == "test":
        dataloaders = datamodule.test_dataloader()
        split = cfg.dataset.split.test
    elif mode in ("submission", "test-on-submission"):
        dataloaders = datamodule.submission_dataloader()
        split = cfg.dataset.split.submission
    outputs = dict()
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on {user}-{session}")

        trainer.test(plmodel, dataloader)

        # save model outputs
        pred_dir = Path(
            cfg.path.logdir.predict.format(user=user, session=session)
        )
        pred_dir.mkdir(parents=True, exist_ok=True)

        for key, arr in plmodel.test_results.items():
            path = Path(pred_dir, f"{key}.npy")
            np.save(path, arr)
            logger.info(f"save {key}[shape={arr.shape}] to {path}")

        key = f"{user}-{session}"
        outputs[key] = {
            "y": plmodel.test_results.get("y"),
            "unixtime": plmodel.test_results.get("unixtime"),
        }
        if mode in ("test", "test-on-submission"):
            outputs[key].update({
                "t_idx": plmodel.test_results.get("t"),
            })

    if mode in ("test", "test-on-submission"):
        # save performance summary
        df_summary = eval_operation_segmentation_wrapper(
            cfg, outputs, OPENPACK_OPERATIONS,
        )
        if mode == "test":
            path = Path(cfg.path.logdir.summary.test)
        elif mode == "test-on-submission":
            path = Path(cfg.path.logdir.summary.submission)

        # NOTE: change pandas option to show tha all rows/cols.
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option("display.width", 200)
        df_summary.to_csv(path, index=False)
        logger.info(f"df_summary:\n{df_summary}")
    elif mode == "submission":
        # make submission file
        metadata = {
            "dataset.split.name": cfg.dataset.split.name,
            "mode": mode,
        }
        submission_dict = construct_submission_dict(
            outputs, OPENPACK_OPERATIONS)
        make_submission_zipfile(submission_dict, logdir, metadata=metadata)


@ hydra.main(version_base=None, config_path="configs",
             config_name="config.yaml")
def main(cfg: DictConfig):
    # DEBUG
    if cfg.debug:
        cfg.dataset.split = optk.configs.datasets.splits.DEBUG_SPLIT
        cfg.path.logdir.rootdir += "/debug"

    print("===== Params =====")
    print(OmegaConf.to_yaml(cfg))
    print("==================")

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode in ("test", "submission", "test-on-submission"):
        test(cfg, mode=cfg.mode)
    else:
        raise ValueError(f"unknown mode [cfg.mode={cfg.mode}]")
    

if __name__ == "__main__":
    main()