import os
import os.path as osp
from argparse import ArgumentParser
# from lit_classifier.all import get_optim_cfg
import mmcv
import torch
from pytorch_lightning import Trainer, seed_everything
from lit_classifier.all import get_trainer
import shutil

def get_exp_by_file(exp_file):
    """
        Copy from https://github.com/Megvii-BaseDetection/YOLOX/blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/exp/build.py
    """
    try:
        import importlib
        import os
        import sys
        sys.path.append(os.path.dirname(exp_file))
        # import ipdb; ipdb.set_trace()
        current_exp = importlib.import_module(
            os.path.basename(exp_file).split(".")[0])
        current_exp = importlib.reload(current_exp)
        exp = current_exp.Exp()
        return exp
    except Exception:
        raise ImportError(
            "{} doesn't contains class named 'Exp'".format(exp_file))

def main(params):
    cfg = get_exp_by_file(params.cfg)
    print(cfg)
    exp_name = osp.basename(params.cfg).split('.')[0]

    data = cfg.get_data_loader()

    model = cfg.get_model(create_lr_scheduler_fn=cfg.get_lr_scheduler(), 
            create_optimizer_fn=cfg.get_optimizer())
    trainer = get_trainer(exp_name,
                          params.devices,
                          distributed=params.devices > 1,
                          max_epochs=cfg.max_epochs)

    # import ipdb; ipdb.set_trace()
    mmcv.mkdir_or_exist(trainer.log_dir)
    shutil.copy(params.cfg, osp.join(trainer.log_dir, osp.basename(params.cfg)))
    if params.verbose:
        print(params.pretty_text)
    trainer.fit(model, data)

if __name__ == '__main__':
    seed_everything(0, workers=True)
    parser = ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--devices',
                        '-d',
                        type=int,
                        default=1,
                        help='Num of gpus')
    parser.add_argument('--max_epochs',
                        '-e',
                        type=int,
                        default=10,
                        help='Max num epochs')
    # parser.add_argument('--batch_size',
    #                     '-b',
    #                     type=int,
    #                     default=64,
    #                     help='Per gpu batch size')
    parser.add_argument('--verbose', action='store_true')

    params = mmcv.Config(parser.parse_args().__dict__)
    main(params)
