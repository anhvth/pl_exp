# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_base_exp.ipynb.

# %% auto 0
__all__ = ['BaseExp']

# %% ../nbs/04_base_exp.ipynb 2
import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import torch
from torch.nn import Module
import pytorch_lightning as pl
from torch.optim.lr_scheduler import _LRScheduler

import os.path as osp


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        # All hyper params should be listed here
        self.accelerator = 'gpu'
        self.max_epochs:int = 100
        self.devices:int = 1
        
        self.schedule_type = 'cosine'
        self.num_lr_cycles = 3
        self.min_lr = 1/100
        self.cycle_decay = 0.7
        
    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(
        self,
    ) -> pl.LightningDataModule:
        pass

    def get_optimizer(self) -> torch.optim.Optimizer:
         return lambda params:torch.optim.Adam(params)


    def get_lr_scheduler(self, train_loader_len=None):
        if train_loader_len is None:
            data = self.get_data_loader()
            data.setup(None)
            #-------
            num_epochs = self.max_epochs
            train_loader = data.train_dataloader()
            train_loader_len = len(train_loader)
        
        if self.schedule_type == 'cosine':
            from ple.lit_model import fn_schedule_cosine_with_warmpup_decay_timm
            create_schedule_fn = fn_schedule_cosine_with_warmpup_decay_timm(
                num_epochs=self.max_epochs,
                num_steps_per_epoch=train_loader_len//self.devices,
                num_epochs_per_cycle=self.max_epochs//self.num_lr_cycles,
                min_lr=self.min_lr,
                cycle_decay=self.cycle_decay,
            )
        elif self.schedule_type == 'linear':
            from ple.lit_model import fn_schedule_linear_with_warmup
            create_schedule_fn = fn_schedule_linear_with_warmup(
                num_epochs=self.trainer.max_epochs,
                num_steps_per_epoch=train_loader_len//self.devices
            )
            
        else:
            raise NotImplementedError
        return create_schedule_fn

        
    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")
    
    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                print(f'Set {k}={v}')
                setattr(self, k, v)
        
    def get_trainer(self):
        from ple.trainer import get_trainer
        return get_trainer(self.exp_name, 
                              max_epochs=self.max_epochs, 
                              gpus=self.devices,
                              trainer_kwargs=dict(
                                  accelerator=self.accelerator,
                              ))
