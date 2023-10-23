# from .imports import *
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import time


class ThroughputLogger(Callback):
    def __init__(self, on_step=True, on_epoch=False) -> None:
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
        log_dir = super().log_dir
        return log_dir.split("/version")[0]



