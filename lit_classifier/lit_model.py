# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_lit_model.ipynb (unless otherwise specified).

__all__ = ['LitModel', 'DEFAULT_HPARAMS', 'get_trainer']

# Cell
import timm



#export
from pytorch_lightning.core.lightning import LightningModule
import torch
from datetime import datetime, timedelta
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from .loss import FocalLoss
import os.path as osp

# Cell
DEFAULT_HPARAMS = {
    "num_workers": 12,
    "image_size": 64,
    "lr": 2e-3,
    "dropout": 0.2,
    "max_epochs": 100,
    "init_lr": 8e-5,
    "num_training_steps": 2000,
}

class LitModel(LightningModule):
    def __init__(self, model, num_classes=None,hparams=DEFAULT_HPARAMS, loss=FocalLoss()):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self._hparams = hparams


    def get_linear_schedule_with_warmup(self,
        optimizer, num_warmup_steps, num_training_steps, init_lr=5e-4,
                                        last_epoch=-1
    ):
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps)) + init_lr
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._hparams["lr"])

        scheduler = {
            "scheduler": self.get_linear_schedule_with_warmup(
                optimizer,
                self._hparams["num_training_steps"] * 0.15,
                self._hparams["num_training_steps"],
                self._hparams["init_lr"],
            ),
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        scores = logits.sigmoid()
        # return dict(scores=scores)
        return scores

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.sigmoid().argmax(1)
        accs = (y == preds).float().mean()


        self.log("val_loss", loss, rank_zero_only=True,
                    on_step=False, on_epoch=True)
        self.log("val_acc", accs, rank_zero_only=True,
                    on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.sigmoid().argmax(1)
        accs = (y == preds).float().mean()

        self.log("training_loss", loss, prog_bar=True, rank_zero_only=True)
        self.log("training_accuracy", accs, prog_bar=True, rank_zero_only=True)
        return loss

# Cell
def get_trainer(exp_name, gpus=1, max_epochs=40, distributed=False,
        monitor=dict(metric="val_acc", mode="max"), save_every_n_epochs=1, save_top_k=5,
    ):


    now = datetime.now() + timedelta(hours=7)
    root_log_dir = osp.join(
            "lightning_logs", exp_name, now.strftime(
                "%b%d_%H_%M_%S")
        )
    filename="{epoch}-{"+monitor["metric"]+":.2f}"

    callback_ckpt = ModelCheckpoint(
        dirpath=osp.join(root_log_dir, "ckpts"),
        monitor=monitor['metric'],mode=monitor['mode'],
        filename=filename,
        save_last=True,
        every_n_epochs=save_every_n_epochs,
        save_top_k=2,
    )

    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    plt_logger = TensorBoardLogger(
        osp.join(root_log_dir, "tb_logs"), version=now.strftime("%b%d_%H_%M_%S")
    )

    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        strategy= "dp" if not distributed else "ddp",
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        logger=plt_logger,
    )
    return trainer