# # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_base_exp.ipynb.


# __all__ = ['BaseExp']


# import ast
# from abc import ABCMeta, abstractmethod
# from typing import Dict

# import pytorch_lightning as pl
# import torch
# from fastcore.all import *
# from tabulate import tabulate
# from torch.nn import Module
# import pprint


# class BaseExp(metaclass=ABCMeta):
#     """
#         import torch.utils.data as td
#         import pytorch_lightning as pl

#         class PLData(pl.LightningDataModule):
#             def __init__(self, **kwargs):
#                 super().__init__()
#                 store_attr()

#             def train_dataloader(self):
#                 # dataset = 
#                 return td.DataLoader(dataset, self.batch_size, num_workers=self.num_workers)

#             def val_dataloader(self):
#                 # dataset = 
#                 return td.DataLoader(dataset, self.batch_size, num_workers=self.num_workers)

#         class TriStageExp(BaseExp):

#             def __init__(self, exp_name='EXPNAME', 
#                          batch_size=64, 
#                          num_workers=2, 
#                          devices=2,
#                          strategy='dp', 
#                          **kwargs):
#                 super().__init__()
#                 store_attr(**kwargs)

#             def get_model(self):
#                 dl = self.get_data_loader().train_dataloader()
#                 sched = fn_schedule_cosine_with_warmpup_decay_timm(
#                     num_epochs=self.max_epochs,
#                     num_steps_per_epoch=len(dl)//self.devices,
#                     num_epochs_per_cycle=self.max_epochs//self.num_lr_cycles,
#                     min_lr=1/100,
#                     cycle_decay=0.7,
#                 )
#                 optim = lambda params:torch.optim.Adam(params)

#                 return MyLit(self.model, create_optimizer_fn=optim,
#                                            create_lr_scheduler_fn=sched)

#             def get_data_loader(self):
#                 return PLData(self.batch_size, num_workers=self.num_workers)

#             def get_trainer(self, **kwargs):
#                 from ple.trainer import get_trainer
#                 return get_trainer(self.exp_name, 
#                                       max_epochs=self.max_epochs, 
#                                       gpus=self.devices,
#                                    strategy=self.strategy,
#                                    **kwargs,

#                                   )
#             exp = TriStageExp( batch_size=10, exp_name='hi', devices=2)
#             print(exp)
#     """

#     def __init__(self):
#         # All hyper params should be listed here
#         self.accelerator = 'gpu'
#         self.max_epochs: int = 100
#         self.devices: int = 1

#         self.schedule_type = 'cosine'
#         self.num_lr_cycles = 3
#         self.min_lr = 1/100
#         self.cycle_decay = 0.7

#     @abstractmethod
#     def get_model(self) -> Module:
#         pass

#     @abstractmethod
#     def get_data_loader(
#         self,
#     ) -> pl.LightningDataModule:
#         pass

#     def get_optimizer(self) -> torch.optim.Optimizer:
#         """
#             Examples:
#                 return lambda params:torch.optim.Adam(params)
#         """
#         pass

#     def get_lr_scheduler(self, train_loader_len=None):
#         if train_loader_len is None:
#             data = self.get_data_loader()
#             data.setup(None)
#             # -------
#             num_epochs = self.max_epochs
#             train_loader = data.train_dataloader()
#             train_loader_len = len(train_loader)

#         if self.schedule_type == 'cosine':
#             from ple.lit_model import \
#                 _lr_function_by_epoch
#             create_schedule_fn = _lr_function_by_epoch(
#                 num_epochs=self.max_epochs,
#                 num_steps_per_epoch=train_loader_len//self.devices,
#                 num_epochs_per_cycle=self.max_epochs//self.num_lr_cycles,
#                 min_lr=self.min_lr,
#                 cycle_decay=self.cycle_decay,
#             )
#         elif self.schedule_type == 'linear':
#             from ple.lit_model import _lr_func_linear_warmup_by_epoch
#             create_schedule_fn = _lr_func_linear_warmup_by_epoch(
#                 num_epochs=self.trainer.max_epochs,
#                 num_steps_per_epoch=train_loader_len//self.devices
#             )

#         else:
#             raise NotImplementedError
#         return create_schedule_fn

#     def __repr__(self):
#         table_header = ["keys", "values"]
#         exp_table = [
#             (str(k), pprint.pformat(v))
#             for k, v in vars(self).items()
#             if not k.startswith("_")
#         ]
#         return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

#     def merge(self, cfg_list):
#         assert len(cfg_list) % 2 == 0
#         for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
#             # only update value with same key
#             if hasattr(self, k):
#                 src_value = getattr(self, k)
#                 src_type = type(src_value)
#                 if src_value is not None and src_type != type(v):
#                     try:
#                         v = src_type(v)
#                     except Exception:
#                         v = ast.literal_eval(v)
#                 print(f'Set {k}={v}')
#                 setattr(self, k, v)

#     def get_trainer(self):
#         from ple.trainer import get_trainer
#         return get_trainer(self.exp_name,
#                            max_epochs=self.max_epochs,
#                            num_gpus=self.devices,
#                            trainer_kwargs=dict(
#                                accelerator=self.accelerator,
#                            ))

#     def plot_lr_sche(self):
#         data = self.get_data_loader()
#         step_per_epoch = len(data.train_dataloader())
#         sched = self.get_lr_scheduler(step_per_epoch)
#         plot_lr_step_schedule(sched, self.lr, self.epochs, step_per_epoch)
