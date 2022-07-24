from argparse import ArgumentParser

import mmcv
import os
import os.path as osp

def get_exp_by_file(exp_file):
    """
        Copy from https://github.com/Megvii-BaseDetection/YOLOX/blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/exp/build.py
    """
    try:
        import sys, os, importlib
        sys.path.append(os.path.dirname(exp_file))
        # import ipdb; ipdb.set_trace()
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        current_exp = importlib.reload(current_exp)
        exp = current_exp.Exp()
        return exp
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))


def get_trainer(exp_name, gpus=1, max_epochs=40, distributed=False,
        monitor=dict(metric="val_acc", mode="max"), save_every_n_epochs=1, save_top_k=1, tqdm_refesh_rate=10,
    trainer_kwargs=dict()):
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from datetime import datetime, timedelta
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer, seed_everything

    now = datetime.now() + timedelta(hours=7)
    
    root_log_dir = osp.join(
            "lightning_logs", exp_name)
    cur_num_exps = len(os.listdir(root_log_dir)) if osp.exists(root_log_dir) else 0
    version = now.strftime(f"{cur_num_exps:02d}_%b%d_%H_%M")
    root_log_dir = osp.join(root_log_dir, version)

    filename="{epoch}-{"+monitor["metric"]+":.2f}"

    callback_ckpt = ModelCheckpoint(
        dirpath=osp.join(root_log_dir, "ckpts"),
        monitor=monitor['metric'],mode=monitor['mode'],
        filename=filename,
        save_last=True,
        every_n_epochs=save_every_n_epochs,
        save_top_k=save_top_k,
    )

    callback_tqdm = TQDMProgressBar(refresh_rate=tqdm_refesh_rate)
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

def main(params):
    cfg = get_exp_by_file(params.cfg)
    exp_name = osp.basename(params.cfg).split('.')[0]
    model = cfg.get_model()
    data = cfg.get_data()
    
    trainer = get_trainer(exp_name)
    
    if params.verbose:
        print(params.pretty_text)
    trainer.fit(model, data)

if __name__ == '__main__':
    seed_everything(0, workers=True)
    parser = ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--verbose', action='store_true')
    # parser = Trainer.add_argparse_args(parser)
    
    params = mmcv.Config(parser.parse_args().__dict__)

    main(params)
