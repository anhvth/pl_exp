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
import ipdb
from loguru import logger
from typing import *
from ple.trainer import TrainingConfig, print_config
from sklearn.metrics import classification_report


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

    def set_trace(self):
        if self.local_rank == 0:
            return ipdb.set_trace

    def configure_optimizers(self):
        return self.optim

    def forward(self, batch):
        x, y = batch
        if isinstance(x, dict):
            try:
                return self.model(**x).logits
            except Exception as e:
                logger.error("I thought the input was hf model but seemslike its not")
                raise NotImplementedError(f"{e}, {type(e)=}")
        else:
            return self.model(x)

    def validation_step(self, batch, batch_idx):
        self.all_logits.append(self(batch))  # Get logits
        self.all_targets.append(batch[1])  # Git y

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(batch)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def gather_dpp_output(self, name):
        """
        Gather the outputs from different distributed processes and concatenate them.

        Parameters:
        - name (str): The name of the attribute (like 'all_logits') to gather across processes.

        Returns:
        - Tensor: Concatenated results from all distributed processes.
        Example usage:
            def on_validation_epoch_end(self):
                gathered_all_logits = self.gather_dpp_output('all_logits')
                gathered_all_targets = self.gather_dpp_output('all_targets')
        """

        # Get the local data based on the given name
        local_data = getattr(self, name)
        if len(local_data) == 0:
            return

        # Convert list of tensors to single tensor (if it's a list)
        if isinstance(local_data, list):
            local_data = torch.cat(local_data)

        # Get world size for distributed processing
        world_size = torch.distributed.get_world_size()

        # Create placeholders for all processes
        gathered_data = [torch.zeros_like(local_data) for _ in range(world_size)]

        # Gather data from all processes
        torch.distributed.all_gather(gathered_data, local_data)

        # Concatenate gathered data
        concatenated_gathered_data = torch.cat(gathered_data, dim=0)

        return concatenated_gathered_data

    def on_validation_epoch_start(self) -> None:
        self.all_logits = []
        self.all_targets = []
        return super().on_validation_epoch_start()


class ClassificationLitModel(AbstractLitModel):
    def on_validation_epoch_end(self):
        gathered_all_logits = self.gather_dpp_output("all_logits")
        gathered_all_targets = self.gather_dpp_output("all_targets")
        if gathered_all_logits is None:
            logger.warning("gathered_all_logits is None!")
            return

        if gathered_all_targets is None:
            logger.warning("gathered_all_targets is None!")
            return

        loss = self.loss_fn(gathered_all_logits, gathered_all_targets)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.local_rank == 0:
            pred_np = gathered_all_logits.argmax(1).cpu().numpy()
            gt_np = gathered_all_targets.cpu().numpy()
            report = classification_report(
                gt_np,
                pred_np,
                zero_division=0,
                output_dict=False,
            )
            logger.info(
                f"Epoch {self.current_epoch}| Global step: {self.global_step}\n"
                + report
            )
            self.logger.experiment.add_text(
                "Classification_Report", report, self.current_epoch
            )
            #====== Report dict
            self.report = classification_report(
                            gt_np,
                            pred_np,
                            zero_division=0,
                            output_dict=True,
                        )
            # Extract macro metrics
            macro_precision = self.report['macro avg']['precision']
            macro_recall = self.report['macro avg']['recall']
            macro_f1_score = self.report['macro avg']['f1-score']
            macro_support = self.report['macro avg']['support']  # optional, as this is just the total count
            accuracy = self.report['accuracy']  # optional, as this is just the total count

            # Log the metrics to TensorBoard
            self.log("val/macro_precision", macro_precision, on_epoch=True, logger=True)
            self.log("val/macro_recall", macro_recall, on_epoch=True, logger=True)
            self.log("val/macro_f1_score", macro_f1_score, on_epoch=True, logger=True)
            self.log("val/macro_support", macro_support, on_epoch=True, logger=True)  # optional
            self.log("val/accuracy", accuracy, on_epoch=True, logger=True) 

            
        self.all_logits.clear()
        self.all_targets.clear()
