import time

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# from .schedulers import 

from torch.utils.data import DataLoader

class ThroughputLogger(Callback):
    def __init__(self, on_step=True, on_epoch=True) -> None:
        self.on_step = on_step
        self.on_epoch = on_epoch
        self.total_samples = 0

        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module) -> None:
        self.start_time = time.time()
        return super().on_train_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: Trainer, *args, **kwargs):
        num_gpus = trainer.num_nodes * trainer.num_devices
        elapsed_time = time.time() - self.start_time
        bz = trainer.train_dataloader.batch_size

        self.total_samples += bz

        one_gpu_speed = self.total_samples / elapsed_time
        all_gpu_speed = one_gpu_speed * num_gpus
        self.log(
            "sps",
            all_gpu_speed,
            prog_bar=True,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
        )


class CustomTensorBoardLogger(TensorBoardLogger):
    @property
    def log_dir(self):
        ld = super().log_dir
        return ld.split("/version")[0]


from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import LambdaLR


from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import LambdaLR

# class CosineWarmupScheduler(Callback):
#     def __init__(
#         self,
#         num_cycles,
#         global_update_steps_total,
#         num_warmup_update_steps=None,  # if not set, use 10% of the steps cycle for warming up
#         init_lr=0.4,
#         min_lr=0.1,
#         cycle_decay=0.6,
#         interval="step",
#     ):
#         super().__init__()
        
#         self.num_cycles = num_cycles
#         self.global_update_steps_total = global_update_steps_total
#         self.global_update_steps_per_cycle = self.global_update_steps_total // num_cycles
        
#         # If not specified, warmup is 10% of the steps in a cycle.
#         if num_warmup_update_steps is None:
#             self.num_warmup_update_steps = int(0.1 * self.global_update_steps_per_cycle)
#         else:
#             self.num_warmup_update_steps = num_warmup_update_steps
            
#         self.init_lr = init_lr
#         self.min_lr = min_lr
#         self.cycle_decay = cycle_decay
#         self.interval = interval

#     def on_train_start(self, trainer:Trainer, pl_module):
#         lr = 1

#         optimizer = trainer.optimizers[0]
        # import ipdb; ipdb.set_trace()
        # schedule = CosineLRScheduler(
        #     optimizer,
        #     t_initial=self.global_update_steps_per_cycle,
        #     lr_min=self.min_lr * lr,
        #     cycle_decay=self.cycle_decay,
        #     cycle_limit=self.num_cycles,
        #     warmup_t=self.num_warmup_update_steps,
        #     warmup_lr_init=self.init_lr * lr,
        # )
        # trainer.lr_scheduler_configs.append(
        #     {"scheduler": schedule, "interval": "step", "frequency": 1}
        # )

import torch.optim as optim

def create_cosine_scheduler(
    optimizer: optim.Optimizer,  # Pass the optimizer as an argument
    data_loader,
    num_epochs,
    accumulate_gradient_steps=1, # 
    training_strategy="ddp",
    num_cycles=2,
    num_warmup_update_steps=None,  # If not set, will use 10% of cycle steps for warmup
    init_lr=0.4, # 40% of the peak 
    min_lr=0.1, # 10% of the peak
    cycle_decay=0.5, # next cycle value is 50% of the peak
    interval="step",#
):
    # Define the parameters for the CosineLRScheduler
    t_initial = num_epochs  # Set the initial number of epochs to the total training epochs
    lr_min = min_lr
    cycle_mul = 1.0
    cycle_limit = num_cycles
    warmup_t = num_warmup_update_steps
    warmup_lr_init = init_lr
    cycle_decay = cycle_decay

    # Create an instance of the CosineLRScheduler
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=t_initial,
        lr_min=lr_min,
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
        warmup_t=warmup_t,
        warmup_lr_init=warmup_lr_init,
    )

    # If warm-up steps are not provided, set to 10% of a cycle's steps
    if num_warmup_update_steps is None:
        cycle_update_steps = scheduler.get_cycle_length(cycles=num_cycles)
        num_warmup_update_steps = int(0.1 * cycle_update_steps)

    return scheduler
