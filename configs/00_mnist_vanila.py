from loguru import logger
import pytorch_lightning as pl
import timm
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

from lit_classifier.lit_model import *
from lit_classifier.loss import FocalLoss

# --------------------Lit model config


class MyLitModel(pl.LightningModule):
    def __init__(self, model, optim_cfg=None, loss=FocalLoss()):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.optim_cfg = optim_cfg
        self.lr = 10e-3

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps,
                                        num_training_steps, init_lr, min_lr,
                                        num_epochs, interval):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                x = (1 - init_lr) * (current_step / num_warmup_steps) + init_lr
                return x
            if interval == 'epoch':
                steps_per_ep = num_training_steps / num_epochs
                current_ep = current_step // steps_per_ep
                current_step = steps_per_ep * current_ep

            total_step = (num_training_steps - num_warmup_steps)
            current_step = current_step - num_warmup_steps
            rt = min_lr + (1 - min_lr) * (1 - current_step / total_step)
            return rt

        return LambdaLR(optimizer, lr_lambda, -1)

    def configure_optimizers(self):
        if self.optim_cfg is None:
            import ipdb; ipdb.set_trace()
            logger.warning('Please add optim cfg and re-init this object')

        if self.optim_cfg['optim'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.optim_cfg["lr"])
        elif self.optim_cfg['optim'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.optim_cfg["lr"],
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0.01,
                                          amsgrad=False)
        else:
            assert isinstance(self.optim_cfg['optim'], torch.optim.Optimizer)
            optimizer = self.optim_cfg['optim']

        scheduler = {
            "scheduler":
            self.get_linear_schedule_with_warmup(
                optimizer, self.optim_cfg["steps"] * 0.15,
                self.optim_cfg["steps"], self.optim_cfg["init_lr"],
                self.optim_cfg["min_lr"], self.optim_cfg["epochs"],
                self.optim_cfg['interval']),
            "interval":
            'step',  # or 'epoch'
            "frequency":
            1,
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

        preds = logits.softmax(1).argmax(1)
        accs = (y == preds).float().mean()

        self.log("val_loss",
                 loss,
                 rank_zero_only=True,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)
        self.log("val_acc",
                 accs,
                 rank_zero_only=True,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.softmax(1).argmax(1)
        accs = (y == preds).float().mean()

        self.log("training_loss",
                 loss,
                 prog_bar=True,
                 rank_zero_only=True,
                 on_epoch=True)
        self.log("training_accuracy",
                 accs,
                 prog_bar=True,
                 rank_zero_only=True,
                 on_epoch=True)
        return loss


#-------------------------------- DATA CONFIG
class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./nbs/",
                 batch_size: int = 32,
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    

    @property
    def train_transform(self):
        from timm.data.transforms_factory import (transforms_imagenet_eval,
                                                transforms_imagenet_train)
        from torchvision import transforms
        def to_rgb(x):
            return x.convert('RGB')

        train_transform = transforms.Compose([
            transforms.Lambda(to_rgb),
            *transforms_imagenet_train(32).transforms,
        ])
        return train_transform

    def setup(self, stage):
        from torchvision.datasets import MNIST
        self.ds_test = MNIST(self.data_dir,
                                train=False,
                                transform=self.train_transform)
        self.ds_predict = MNIST(self.data_dir,
                                   train=False,
                                   transform=self.train_transform)
        ds_full = MNIST(self.data_dir,
                           train=True,
                           transform=self.train_transform)
        self.ds_train, self.ds_val = random_split(
            ds_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.ds_predict,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


class Exp:
    def get_model(self, **kwargs):
        model = timm.create_model('resnet18', True, num_classes=10)
        return MyLitModel(model, **kwargs)

    def get_data(self, batch_size, **kwargs):
        return DataModule(batch_size=batch_size, **kwargs)
