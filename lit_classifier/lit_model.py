# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_lit_model.ipynb (unless otherwise specified).

__all__ = ['get_optim_cfg', 'LitModel', 'BinLitModel', 'load_lit_state_dict', 'get_trainer']

# Cell
from loguru import logger
from pytorch_lightning.core.lightning import LightningModule
import torch
from datetime import datetime, timedelta
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from .loss import FocalLoss, BinaryFocalLoss
import os.path as osp

# Cell

#export
def get_optim_cfg(epochs, steps_per_ep, lr=1e-3, init_lr=0.5, min_lr=0.2, interval='step', optim='Adam'):
    steps = epochs*steps_per_ep
    return dict(lr=lr, init_lr=init_lr, min_lr=min_lr, steps=steps, epochs=epochs, interval=interval, optim=optim)


# Cell
# def lr_lambda(current_step: int):
#     if current_step < num_warmup_steps:
#         x = (1-init_lr)*(current_step / num_warmup_steps)+init_lr
#         return x
#     if interval=='epoch':
#         steps_per_ep = num_training_steps / num_epochs
#         current_ep = current_step // steps_per_ep
#         current_step = steps_per_ep*current_ep

#     total_step = (num_training_steps-num_warmup_steps)
#     current_step = current_step-num_warmup_steps
#     rt = min_lr+(1-min_lr)*(1-current_step/total_step)
#     return rt

from torch.optim.lr_scheduler import LambdaLR
class LitModel(LightningModule):
    def __init__(self, model, optim_cfg=None, loss=FocalLoss()):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.optim_cfg = optim_cfg
        self.lr = 10e-3

    # def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, init_lr, min_lr, num_epochs, interval):
    #     return LambdaLR(optimizer, lr_lambda, -1)

    def configure_optimizers(self):
        if self.optim_cfg is None:
            logger.warning('Please add optim cfg and re-init this object')
            return
        if self.optim_cfg['optim'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_cfg["lr"])
        elif self.optim_cfg['optim'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_cfg["lr"],
                                          betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        else:
            assert isinstance(self.optim_cfg['optim'], torch.optim.Optimizer)
            optimizer = self.optim_cfg['optim']

        # scheduler = {
        #     "scheduler": self.get_linear_schedule_with_warmup(
        #         optimizer,
        #         self.optim_cfg["steps"] * 0.15,
        #         self.optim_cfg["steps"],
        #         self.optim_cfg["init_lr"],
        #         self.optim_cfg["min_lr"],
        #         self.optim_cfg["epochs"],
        #         self.optim_cfg['interval']
        #     ),
        #     "interval": 'step',  # or 'epoch'
        #     "frequency": 1,
        # }

        return [optimizer]#, [scheduler]

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

        preds = logits.softmax(1).argmax(1)
        accs = (y == preds).float().mean()


        self.log("val_loss", loss, rank_zero_only=True, prog_bar=True,
                    on_step=False, on_epoch=True)
        self.log("val_acc", accs, rank_zero_only=True, prog_bar=True,
                    on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.softmax(1).argmax(1)
        accs = (y == preds).float().mean()

        self.log("training_loss", loss, prog_bar=True, rank_zero_only=True, on_epoch=True)
        self.log("training_accuracy", accs, prog_bar=True, rank_zero_only=True, on_epoch=True)
        return loss


class BinLitModel(LitModel):
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x).reshape(-1)
        y = y.reshape(logits.shape)
        loss = self.loss_fn(logits, y)

        preds = logits.sigmoid() > 0.5
        accs = (y == preds).float().mean()


        self.log("val_loss", loss, rank_zero_only=True,
                    on_step=False, on_epoch=True)
        self.log("val_acc", accs, rank_zero_only=True,
                    on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x).reshape(-1)
        y = y.reshape(logits.shape)
        loss = self.loss_fn(logits, y)

        preds = logits.sigmoid() > 0.5
        accs = (y == preds).float().mean()

        self.log("training_loss", loss, prog_bar=True, rank_zero_only=True)
        self.log("training_accuracy", accs, prog_bar=True, rank_zero_only=True)
        return loss

# from lit_classifier.lit_model import LitModel
# PLitModel = persistent_class(LitModel)

# Cell
def load_lit_state_dict(ckpt_path):
    st = torch.load(ckpt_path)['state_dict']
    out_st = {}
    for k, v in st.items():
        out_st[k.replace('model.', '')] = v
    return out_st

# Cell
def get_trainer(exp_name, gpus=1, max_epochs=40, distributed=False,
        monitor=dict(metric="val_acc", mode="max"), save_every_n_epochs=1, save_top_k=1, use_version=True,
    trainer_kwargs=dict()):

    now = datetime.now() + timedelta(hours=7)

    root_log_dir = osp.join(
            "lightning_logs", exp_name)
    cur_num_exps = len(os.listdir(root_log_dir)) if osp.exists(root_log_dir) else 0
    version = now.strftime(f"{cur_num_exps:02d}_%b%d_%H_%M")
    if use_version:
        root_log_dir = osp.join(root_log_dir, version)
        logger.info('Root log directory: {}'.format(root_log_dir))
    filename="{epoch}-{"+monitor["metric"]+":.2f}"

    callback_ckpt = ModelCheckpoint(
        dirpath=osp.join(root_log_dir, "ckpts"),
        monitor=monitor['metric'],mode=monitor['mode'],
        filename=filename,
        save_last=True,
        every_n_epochs=save_every_n_epochs,
        save_top_k=save_top_k,
    )

    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    plt_logger = TensorBoardLogger(
        osp.join(root_log_dir, "tb_logs"), version=version
    )

    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        strategy= "dp" if not distributed else "ddp",
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        logger=plt_logger,**trainer_kwargs,
    )
    return trainer