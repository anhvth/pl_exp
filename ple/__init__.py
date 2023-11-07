# # __version__ = "0.0.5"
# import os
# from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# # from fastcore.all import *
# from loguru import logger
from pytorch_lightning import LightningDataModule, LightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ple.lit_model import AbstractLitModel, PLDataModule, ClassificationLitModel
from ple.schedulers import get_cyclic_cosine_lr
from ple.trainer import TrainingConfig, get_trainer_from_config, print_config

from .utils import load_ckpt_inner, to_device