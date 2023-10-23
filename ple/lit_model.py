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
    
    # def __getattr__(self, name: str) -> Any:
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         if hasattr(self, "config") and hasattr(self.config, name):
    #             return getattr(self.config, name)
    #         raise

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optim

        if not self.config.train_data_len is not None:
            logger.info("Not enough info to initialize scheduler")
            raise ValueError(
                "Incomplete information for scheduler initialization")

        lr_func_by_step = get_cyclic_cosine_lr(self.config)
        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_func_by_step, -1),
            "interval": "step",  # step-based update
            "frequency": self.config.grad_accumulate_steps,
        }

        return [optimizer], [scheduler]

    def compute_predictions(self, logits):
        return logits.softmax(1).argmax(1)

    def log_metrics(self, metrics, prog_bar=True, on_epoch=True):
        for key, value in metrics.items():
            self.log(key,
                     value,
                     prog_bar=prog_bar,
                     rank_zero_only=True,
                     on_epoch=on_epoch)

    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = self.loss_fn(pred, batch[1])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = self.loss_fn(pred, batch[1])
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss
