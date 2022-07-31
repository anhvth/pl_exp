from loguru import logger
import pytorch_lightning as pl
import timm
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

from lit_classifier.all import *

# --------------------Lit model config

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


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.lr = 0.15
        self.batch_size = 128
        self.num_lr_cycles = 3
        self.max_epochs = 20


    def get_optimizer(self):
        create_optim_fn = lambda params: torch.optim.Adam(params, lr=self.lr)
        return create_optim_fn

    def get_lr_scheduler(self):
        data = self.get_data_loader()
        data.setup(None)
        #-------
        num_epochs = self.max_epochs
        schedule_type = "cosine"
        train_loader = data.train_dataloader()
        
        if schedule_type == 'cosine':
            create_schedule_fn = fn_schedule_cosine_with_warmpup_decay_timm(
                num_epochs=self.max_epochs,
                num_steps_per_epoch=len(train_loader),
                num_epochs_per_cycle=self.max_epochs//self.num_lr_cycles
            )
        elif schedule_type == 'linear':
            create_schedule_fn = fn_schedule_linear_with_warmup(
                num_epochs=self.trainer.max_epochs,
                num_steps_per_epoch=len(train_loader)
            )
        else:
            raise NotImplementedError
        return create_schedule_fn

    def get_model(self, **kwargs):
        model = timm.create_model('mobilenetv2_035', True, num_classes=10)
        return LitModel(model, **kwargs)

    def get_data_loader(self,**kwargs):
        if not hasattr(self, 'data'):
            self.data = DataModule(batch_size=self.batch_size, **kwargs)
        return self.data

if __name__ == '__main__':
    exp = Exp()
    print(exp)