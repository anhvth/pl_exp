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

DL_TRAIN = torch.utils.data.DataLoader(val_ds, BZ, num_workers=NUM_WORKERS, shuffle=True)
DL_VAL = torch.utils.data.DataLoader(val_ds, BZ, num_workers=NUM_WORKERS, shuffle=False)

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
        x = self.layers(x.flatten(1))
        return x

model = SimpleCNN()

# ---- LEARNING SETUP ----
optim = lambda params: torch.optim.Adam(params, lr=LR)



import pytorch_lightning as pl
import torch.nn as nn
import torch

class CustomLitModel(AbstractLitModel):

    def validation_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = self.loss_fn(pred, batch[1])
        self.log('val/loss', loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = self.loss_fn(pred, batch[1])
        self.log('train/loss', loss, prog_bar=True)
        return loss

# Usage
lit = CustomLitModel(
    model=model,
    lr=LR,
    loss_fn=nn.CrossEntropyLoss(),
    train_loader=DL_TRAIN,
    num_epochs=EPOCHS, 
    grad_accumulate_steps=GRAD_ACCUMULATE_STEPS,
    optim = torch.optim.Adam(model.parameters(), 1e-5)
)    

# ---- TRAINING ----
trainer = get_trainer(
    name=EXP_NAME,
    max_epochs=EPOCHS,
    num_gpus=GPUS,
    overfit_batches=0.0, # You can adjust this if needed
    monitor={'metric': 'val/loss', 'mode': 'min'},
    strategy=STRATEGY,
)

# import ipdb; ipdb.set_trace()
trainer.fit(lit, DL_TRAIN, DL_VAL)
