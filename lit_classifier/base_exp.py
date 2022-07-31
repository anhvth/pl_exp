# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_exp.ipynb.

# %% auto 0
__all__ = ['BaseExp']

# %% ../nbs/04_exp.ipynb 2
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import torch
from torch.nn import Module
import pytorch_lightning as pl
from torch.optim.lr_scheduler import _LRScheduler


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10
        self.max_epochs = 20

    
    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(
        self, 
    )->pl.LightningDataModule:
        pass

    @abstractmethod
    def get_optimizer() -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> _LRScheduler:
        pass

#     @abstractmethod
#     def get_evaluator(self):
#         pass

#     @abstractmethod
#     def eval(self, model, evaluator, weights):
#         pass

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")
