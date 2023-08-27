import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional


import hydra
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from scipy.special import softmax
from model.models import *
import matplotlib.pyplot as plt
import seaborn as sns
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from openpack_toolkit import OPENPACK_OPERATIONS

class MyModelLM(optorch.lightning.BaseLightningModule):

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        """ Initalize your model
        
        Generate an instance of the model you defined in the above.
        Uncomment the target model.
        """
        # !! Edit Here !!
        # NOTE: Please select the model you want to use!
        #model = DeepConvLstmV1()
        # model = DeepConvLstm()
        #model = DeepConvLstmV3()        
        model = DeepConvLSTMSelfAttn()
        

        summary(model, input_size=(32, 46, 1800, 1))
        return model
    
    
    def init_criterion(self, cfg: DictConfig):
        """Initialize loss function
        """
        ignore_cls = [(i, c) for i, c in enumerate(cfg.dataset.classes.classes) if c.is_ignore]
        
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_cls[-1][0]
        )
        return criterion

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of training loop. Get mini-batch and return loss.
        """
        x = batch["x"].to(device=self.device, dtype=torch.float)        
        t = batch["t"].to(device=self.device, dtype=torch.long)
        y_hat = self(x).squeeze(3)

        #print("Input tensor size:", x.shape)
        #print("Output tensor size:", y_hat.shape)
        #print("Size of tensor after layer 1:", self.conv.weight.shape)
        #print("Size of tensor after layer 2:", self.lstm.weight.shape)
        #print("Size of tensor after layer 3:", self.attention.weight.shape)

        loss = self.criterion(y_hat, t)
        acc = self.calc_accuracy(y_hat, t)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of inference step. Get mini-batch and return model outputs.
        """
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        ts_unix = batch["ts"]

        y_hat = self(x).squeeze(3)

        outputs = dict(t=t, y=y_hat, unixtime=ts_unix)
        return outputs
    
class SplitDataModelLM(optorch.lightning.BaseLightningModule):

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:               
        model = CSNetWithFusion()
        #summary(model, input_size=(32, 46, 1800, 1))
        return model
    
    
    def init_criterion(self, cfg: DictConfig):
        """Initialize loss function
        """
        ignore_cls = [(i, c) for i, c in enumerate(cfg.dataset.classes.classes) if c.is_ignore]
        
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_cls[-1][0]
        )
        return criterion

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of training loop. Get mini-batch and return loss.
        """
        print("train step")
        x_imu = batch["x_imu"].to(device=self.device, dtype=torch.float)
        x_keypoints = batch["x_keypoints"].to(device=self.device, dtype=torch.float)    
        x_e4 = batch["x_e4"].to(device=self.device, dtype=torch.float)            
        t = batch["label_imu"].to(device=self.device, dtype=torch.long)

        #print("Input imu size:", x_imu.shape)
        #print("Input kp size:", x_keypoints.shape)
        #print("Input e4 size:", x_e4.shape)
        #print("Input labels size:", t.shape)
        #print("Output tensor size:", y_hat.shape)
        #print("Size of tensor after layer 1:", self.conv.weight.shape)
        #print("Size of tensor after layer 2:", self.lstm.weight.shape)
        #print("Size of tensor after layer 3:", self.attention.weight.shape)

        y_hat = self([x_imu, x_keypoints, x_e4])
        #print("y_hat shape:", y_hat.shape)

        loss = self.criterion(y_hat, t)
        acc = self.calc_accuracy(y_hat, t)     

        self.log("train_loss",loss, on_epoch=True, on_step=False)
        self.log("train_acc",acc, on_epoch=True, on_step=False)   

        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of inference step. Get mini-batch and return model outputs.
        """
        x_imu = batch["x_imu"].to(device=self.device, dtype=torch.float)
        x_keypoints = batch["x_keypoints"].to(device=self.device, dtype=torch.float)    
        x_e4 = batch["x_e4"].to(device=self.device, dtype=torch.float)            
        t = batch["label_imu"].to(device=self.device, dtype=torch.long)
        ts = batch["times_imu"]
        y_hat = self([x_imu, x_keypoints, x_e4])

        outputs = dict(t=t, y=y_hat,unixtime=ts)
        return outputs
    

class SensorFusionModelLM(optorch.lightning.BaseLightningModule):

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:               
        model = CSNetWithSensorFusion()
        return model
    
    
    def init_criterion(self, cfg: DictConfig):
        """Initialize loss function
        """
        ignore_cls = [(i, c) for i, c in enumerate(cfg.dataset.classes.classes) if c.is_ignore]
        
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_cls[-1][0]
        )
        return criterion

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of training loop. Get mini-batch and return loss.
        """
        print("train step")
        x = batch["x"].to(device=self.device, dtype=torch.float)        
        t = batch["t"].to(device=self.device, dtype=torch.long)

        #print("Input imu size:", x_imu.shape)
        #print("Input kp size:", x_keypoints.shape)
        #print("Input e4 size:", x_e4.shape)
        #print("Input labels size:", t.shape)
        #print("Output tensor size:", y_hat.shape)
        #print("Size of tensor after layer 1:", self.conv.weight.shape)
        #print("Size of tensor after layer 2:", self.lstm.weight.shape)
        #print("Size of tensor after layer 3:", self.attention.weight.shape)

        y_hat = self(x)
        #print("y_hat shape:", y_hat.shape)

        loss = self.criterion(y_hat, t)
        acc = self.calc_accuracy(y_hat, t)     

        self.log("train_loss",loss, on_epoch=True, on_step=False)
        self.log("train_acc",acc, on_epoch=True, on_step=False)   

        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of inference step. Get mini-batch and return model outputs.
        """
        x = batch["x"].to(device=self.device, dtype=torch.float)        
        t = batch["t"].to(device=self.device, dtype=torch.long)
        ts = batch["ts"]
        y_hat = self(x)

        outputs = dict(t=t, y=y_hat,unixtime=ts)
        return outputs
    

class IndividualModelForDecisionLM(optorch.lightning.BaseLightningModule):

    def __init__(self, model_type="", cfg: DictConfig = None) -> None:
        if (model_type):
            self.model_type = model_type
        else:
            self.model_type = cfg.model_type         
        super().__init__(cfg)

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:  
        channels, reshape, seq_length = get_parameters_by_type(self.model_type)             
        model = CSNetIndividualModelForDecision(in_ch=channels, reshape_len=reshape,out_features=seq_length)
        return model
    
    
    def init_criterion(self, cfg: DictConfig):
        """Initialize loss function
        """
        ignore_cls = [(i, c) for i, c in enumerate(cfg.dataset.classes.classes) if c.is_ignore]
        
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_cls[-1][0]
        )
        return criterion

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of training loop. Get mini-batch and return loss.
        """
        x_data = "x_"+self.model_type
        x = batch[x_data].to(device=self.device, dtype=torch.float)        
        label = "label_"+self.model_type                   
        t = batch[label].to(device=self.device, dtype=torch.long)

        #print("Input imu size:", x_imu.shape)
        #print("Input kp size:", x_keypoints.shape)
        #print("Input e4 size:", x_e4.shape)
        #print("Input labels size:", t.shape)
        #print("Output tensor size:", y_hat.shape)
        #print("Size of tensor after layer 1:", self.conv.weight.shape)
        #print("Size of tensor after layer 2:", self.lstm.weight.shape)
        #print("Size of tensor after layer 3:", self.attention.weight.shape)

        y_hat = self(x)     

        loss = self.criterion(y_hat, t)
        acc = self.calc_accuracy(y_hat, t)     

        self.log("train_loss",loss, on_epoch=True, on_step=False)
        self.log("train_acc",acc, on_epoch=True, on_step=False)   

        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Definition of inference step. Get mini-batch and return model outputs.
        """
        x_data = "x_"+self.model_type
        x = batch[x_data].to(device=self.device, dtype=torch.float)        
        label = "label_"+self.model_type                   
        t = batch[label].to(device=self.device, dtype=torch.long)
        times_data = "times_"+self.model_type
        ts = batch[times_data]
        y_hat = self(x)

        outputs = dict(t=t, y=y_hat,unixtime=ts)
        return outputs

def get_parameters_by_type(model_type):
    if (model_type == "imu"):
        return 12, 225, 1800
    if (model_type == "keypoints"):
        return 34, 113, 900
    if (model_type == "e4"):
        return 6, 240, 1920
    
    return 0,0
    
        
        

