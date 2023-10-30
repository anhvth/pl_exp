import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule, LightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ple.schedulers import get_cyclic_cosine_lr
from ple.trainer import TrainingConfig, get_trainer_from_config

from loguru import logger
from typing import *
from ple.trainer import TrainingConfig, print_config


class PLDataModule(LightningDataModule):

    def __init__(
        self,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        collate_fn=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.config = config
        self.config.train_data_len = len(train_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.data_num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.data_num_workers,
            collate_fn=self.collate_fn,
        )


class AbstractLitModel(LightningModule):

    def __init__(
        self,
        model: Module,
        optim: Optimizer,
        loss_fn: Callable,
        config: TrainingConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.optim = optim
    
    def forward(self, batch):
        x, y = batch
        return self.model(**x).logits

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(batch)
        loss = self.loss_fn(pred, y)
        # Calculate accuracy and F1 score
        acc = self.accuracy(pred.argmax(dim=1), y)
        f1 = self.f1(pred.argmax(dim=1), y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(batch)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss
