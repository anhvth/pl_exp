from ple import *
from torchvision import transforms, datasets
import torch.nn.functional as F

# ---- CONFIGURATION ----
def get_config():
    return TrainingConfig(
        strategy="ddp",
        exp_name="mnist",
        val_check_interval=None,
        monitor_metric="val_loss",
        batch_size=32,
        num_gpus=8,
        ckpt_save_top_k=1,
    )

# ---- DATA SETUP ----
def get_mnist_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    val_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    return train_ds, val_ds

# ---- MODEL SETUP ----
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(784, 256), nn.ReLU(), torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.layers(x.flatten(1))
        return x

# ---- TRAINING ----
def train(train_ds, val_ds, model, config):
    datamodule = PLDataModule(train_ds, val_ds, config)
    loss_fn = nn.CrossEntropyLoss()
    lit = ClassificationLitModel(
        model=model,
        loss_fn=loss_fn,
        config=config,
    ).cpu()
    trainer = get_trainer_from_config(config=config)
    if trainer.local_rank == 0:
        print_config(config)
    trainer.fit(lit, datamodule=datamodule)

if __name__ == '__main__':
    config = get_config()
    train_ds, val_ds = get_mnist_data()
    model = SimpleCNN()
    train(train_ds, val_ds, model, config)
