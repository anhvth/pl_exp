

import torch
def load_ckpt_inner(model, ckpt_path, strict=False, infer_mode=True):
    ckpt = torch.load(ckpt_path)
    state_dict = {}
    for k, v in ckpt['state_dict'].items():
        state_dict[k[6:]] = v
    res =  model.load_state_dict(state_dict, strict=strict)
    print(f'{infer_mode=}')
    if infer_mode:
        model.requires_grad_(False).eval()
    return res