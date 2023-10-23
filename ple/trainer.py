# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_trainer.ipynb.

import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Union

from fastcore.script import *
from fastcore.utils import *


__all__ = ["is_interactive", "get_trainer", "get_rank", "get_exp_by_file", "train"]

import os.path as osp

import tabulate
import torch
from loguru import logger
from pytorch_lightning import Trainer

from .custom_callbacks import *


def is_interactive():
    return "get_ipython" in dir()


def print_config(config):
    config_dict = asdict(config)
    grid = tabulate.tabulate(
        list(config_dict.items()), headers=["Parameter", "Value"], tablefmt="fancy_grid"
    )
    logger.info("\n" + grid)


from dataclasses import dataclass
from typing import Dict, Union, Any

from dataclasses import dataclass
from typing import Dict, Union, Any


@dataclass
class TrainingConfig:
    # Training parameters
    lr: float = 1e-4
    num_epochs: int = 4
    batch_size: int = 4
    grad_accumulate_steps: int = 2
    num_gpus: int = 1
    strategy: str = "ddp"
    overfit_batches: float = 0.0
    train_data_len: int = None
    precision: str = "16-mixed"
    accelerator: str = "gpu"

    # Scheduler parameters
    sched_num_cycle: int = 2
    sched_warm_up_step: float = 0.1
    sched_cycle_decay: float = 0.5
    sched_min_lr: float = 0.1
    sched_init_lr: float = 0.4

    # Validation parameters

    data_num_workers: int = 1

    # Logging/Monitoring parameters
    exp_name: str = "exp_name"
    refresh_rate: int = 10
    find_unused_parameters: bool = True

    # Checkpoint parameters
    val_check_interval: int = None
    ckpt_save_last: bool = True
    ckpt_save_top_k: int = -1
    ckpt_every_n_epochs: int = 1
    ckpt_every_n_train_steps: int = None

    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    def __post_init__(self):
        self.global_batch_size = (
            self.num_gpus * self.batch_size * self.grad_accumulate_steps
        )

        # Override ckpt_every_n_epochs with val_check_interval if the latter is not None
        if (
            self.ckpt_every_n_train_steps is not None
            and self.monitor_metric is not None
        ):
            self.val_check_interval = self.ckpt_every_n_train_steps
            self.ckpt_every_n_epochs = None
            # There is a provided metrics so set topk to 1
            if self.ckpt_save_top_k == -1:
                self.ckpt_save_top_k = 1


from dataclasses import asdict


def get_trainer_from_config(config: TrainingConfig, **kwargs) -> Trainer:
    if not torch.cuda.is_available():
        config.num_gpus = 1
        config.accelerator = "cpu"
        logger.warning("No GPU available, using CPU instead, gpus=1, accelerator='cpu'")

    callbacks = []
    plt_logger = None

    if config.exp_name:
        rld = osp.join("lightning_logs", config.exp_name)
        cur_num_exps = len(os.listdir(rld)) if osp.exists(rld) else 0
        version = f"{cur_num_exps:02d}"
        root_log_dir = osp.join("lightning_logs", config.exp_name, version)
        metric = config.monitor_metric
        metric = metric if metric is None else metric.replace("/", "_")
        if (
            config.ckpt_every_n_epochs is None
            and config.ckpt_every_n_train_steps is not None
        ):
            filename = f"{{step}}_{{{metric}:0.2f}}"
        elif config.monitor_metric is not None:
            filename = f"{{epoch}}_{{{metric}:0.2f}}"
        else:
            filename = None

        callback_ckpt = ModelCheckpoint(
            dirpath=osp.join(root_log_dir),
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            filename=filename,
            save_last=config.ckpt_save_last,
            save_top_k=config.ckpt_save_top_k,
            every_n_train_steps=config.ckpt_every_n_train_steps,
            every_n_epochs=config.ckpt_every_n_epochs,
        )

        plt_logger = CustomTensorBoardLogger(osp.join(root_log_dir))
        callbacks.append(callback_ckpt)

    callbacks += [
        TQDMProgressBar(refresh_rate=config.refresh_rate),
        LearningRateMonitor(logging_interval="step"),
        ThroughputLogger(),
    ]

    if not config.strategy:
        if is_interactive():
            logger.info(
                f"gpus={config.num_gpus}, Interactive mode, force strategy=auto"
            )
            config.strategy = "auto"
        elif config.num_gpus < 2:
            logger.info(f"gpus={config.num_gpus}, , force strategy=dp")
            config.strategy = "auto"
        else:
            config.strategy = "ddp"

    if config.strategy == "ddp":
        from pytorch_lightning.strategies.ddp import DDPStrategy

        strategy = DDPStrategy(find_unused_parameters=config.find_unused_parameters)
    else:
        strategy = config.strategy
    # Convert the dataclass to a dictionary and unpack its values into the Trainer constructor.
    trainer = Trainer(
        accelerator=config.accelerator,
        devices=config.num_gpus,
        max_epochs=config.num_epochs,
        strategy=strategy,
        val_check_interval=config.val_check_interval,
        callbacks=callbacks,
        logger=plt_logger,
        precision=config.precision,
    )
    return trainer


def get_rank() -> int:
    import torch.distributed as dist

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_exp_by_file(exp_file):
    """
    Params:
    exp_file: Path to exp

    """
    try:
        import importlib
        import os
        import sys

        sys.path.append(os.path.dirname(exp_file))
        # import ipdb; ipdb.set_trace()
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        current_exp = importlib.reload(current_exp)
        exp = current_exp.Exp()
        return exp
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))


@call_parse
def train(
    cfg_path: Param("Path to config"),
    devices: Param("GPUS indices", default=1, type=int),
    opts: Param("Additional configs", default="", type=str, required=False),
):
    cfg = get_exp_by_file(cfg_path)
    if len(opts):
        cfg.merge(opts.replace("=", " ").split(" "))
    cfg.devices = devices

    data = cfg.get_data_loader()

    model = cfg.get_model(
        create_lr_scheduler_fn=cfg.get_lr_scheduler(),
        create_optimizer_fn=cfg.get_optimizer(),
    )
    trainer = cfg.get_trainer(devices)
    try:
        trainer.fit(model, data)
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        if get_rank() == 0:
            out_path = osp.join(trainer.log_dir, osp.basename(cfg_path))
            logger.info("cp {} {}", cfg_path, out_path)
            shutil.copy(cfg_path, out_path)
