from ple.all import *
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import timm
from avcv.all import *
import pytorch_lightning as pl

model = timm.create_model('mobilenetv2_035', pretrained=False, num_classes=10, in_chans=1)

train_ds = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
val_ds = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
lit = LitForClassification(model, train_ds, val_ds, data_batch_size=2, lr_num_cycles=2, data_num_workers=4, lr_update_interval='epoch')
                               
#---------------- Train
trainer = get_trainer('mnist', max_epochs=30, monitor=dict(metric="val_loss", mode="min"),
                strategy='ddp', gpus=1, overfit_batches=0.001, num_nodes=1)
trainer.fit(lit)

