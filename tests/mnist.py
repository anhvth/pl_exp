import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
from ple import *

# ---- HYPERPARAMETERS ----
EPOCHS = 3
LR = 1e-4
BZ = 1
GRAD_ACCUMULATE_STEPS = 8
GPUS = 1
NUM_WORKERS = 1
STRATEGY = 'auto'
EXP_NAME = 'exp_name'

# ---- DATA SETUP ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Train and Validation Split
train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_length = int(0.99 * len(train_ds))
val_length = len(train_ds) - train_length
train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_length, val_length])

dl_train = torch.utils.data.DataLoader(val_ds, BZ, num_workers=NUM_WORKERS, shuffle=True)
dl_val = torch.utils.data.DataLoader(val_ds, BZ, num_workers=NUM_WORKERS, shuffle=False)

# ---- MODEL DEFINITION ----
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(784, 256),
            nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = x.flatten(1)
        x = self.layers(x)
        return x

model = SimpleCNN()

# ---- LEARNING SETUP ----
optim = lambda params: torch.optim.Adam(params, lr=LR)


# sched = _lr_function_by_epoch(
#     num_epochs=EPOCHS,
#     num_steps_per_epoch=len(dl_train),
#     num_epochs_per_cycle=EPOCHS // 2,
#     min_lr=1/100,
#     cycle_decay=0.7
# )

import pytorch_lightning as pl
import torch.nn as nn
import torch

class LitModel(AbstractLitModel):
    def validation_step(self, batch, batch_idx):
        
        pred = self(batch[0])
        loss = self.loss_fn(pred, batch[1])
        self.log('val/loss', loss, prog_bar=True)
    def training_step(self, batch, batch_idx):
        print(self.global_step)
        pred = self(batch[0])
        loss = self.loss_fn(pred, batch[1])
        self.log('train/loss', loss, prog_bar=True, )
        return loss
# print(self.global_step)

# Usage
lit = LitModel(
    model=model,
    lr=1e-5, # You can specify this if you want to override the default
    loss_fn=nn.CrossEntropyLoss(),
    train_loader=dl_train,
    num_epochs=EPOCHS, 
    grad_accumulate_steps=GRAD_ACCUMULATE_STEPS,
    optim = torch.optim.Adam(model.parameters(), 1e-5)
)    

# ---- TRAINING ----
trainer = get_trainer(
    EXP_NAME, EPOCHS, num_gpus=GPUS, overfit_batches=0.0,
    monitor={'metric': 'val/loss', 'mode': 'min'}, 
    strategy=STRATEGY, 
)
# import ipdb; ipdb.set_trace()
trainer.fit(lit, dl_train, dl_val)
