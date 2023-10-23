from ple import *
from dataclasses import asdict, dataclass, field

from torchvision import transforms, datasets


torch.set_float32_matmul_precision("medium")
train_config = TrainingConfig(
    strategy="ddp",
    val_check_interval=100,
    monitor_metric='val_loss',
    num_gpus=1,
)
loss_fn = nn.CrossEntropyLoss()
# ---- DATA SETUP ----
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
# Train and Validation Split
train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
val_length = 16 * 100
train_length = len(train_ds) - val_length
train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_length, val_length])
train_ds = val_ds
datamodule = PLDataModule(train_ds, val_ds, train_config)


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(784, 256), nn.ReLU(), torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.layers(x.flatten(1))
        return x


model = SimpleCNN()

# Usage
optim = torch.optim.Adam(model.parameters(), 1e-5)
lit = AbstractLitModel(
    model=model,
    loss_fn=loss_fn,
    config=train_config,
    optim=optim,
).cpu()

# ---- TRAINING ----
trainer = get_trainer_from_config(
    config=train_config,
)

if trainer.local_rank == 0:
    print_config(train_config)


trainer.fit(lit, datamodule=datamodule)
