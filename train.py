from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
import mmcv

def get_exp_by_file(exp_file):
    """
        Copy from https://github.com/Megvii-BaseDetection/YOLOX/blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/exp/build.py
    """
    try:
        import sys, os, importlib
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def main(args):
    cfg = get_exp_by_file(args.cfg)
    import ipdb; ipdb.set_trace()
    model = cfg.get_model()
    data = cfg.get_data()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data)

if __name__ == '__main__':
    seed_everything(0, workers=True)
    parser = ArgumentParser()
    parser.add_argument('cfg')
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
