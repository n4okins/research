# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from pathlib import Path


__device__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__file__ = Path(locals().get("__file__", "main.py")).resolve()
__root__ = __file__.parents[2]
__data__ = __root__ / "data"

print(f"{__device__=}\n{__file__=}\n{__root__=}\n{__data__=}")


# %%
# Load Mnist dataset


train_dataset = torchvision.datasets.MNIST(
    root=str(__data__), train=True, download=True, transform=transforms.ToTensor()
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
)


def sampling_plot():
    _train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    x, y = next(iter(_train_dataloader))
    print(f"input shape: {x.shape}")
    print(f"labels: {y}")
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap="gray")
    plt.axis("off")
    plt.show()


sampling_plot()
# %%


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


def sampling_corrupt_plot():
    _train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    x, y = next(iter(_train_dataloader))
    print(f"input shape: {x.shape}")
    print(f"labels: {y}")
    amount = torch.linspace(0, 1, x.shape[0])
    print(f"{amount=}")
    plt.imshow(torchvision.utils.make_grid(corrupt(x, amount))[0], cmap="gray")
    plt.axis("off")
    plt.show()


sampling_corrupt_plot()

# %%


class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_size=32):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, hidden_size, 5, padding=2),
                nn.Conv2d(hidden_size, hidden_size * 2, 5, padding=2),
                nn.Conv2d(hidden_size * 2, hidden_size * 2, 5, padding=2),
            ]
        )
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Conv2d(hidden_size * 2, hidden_size * 2, 5, padding=2),
                nn.Conv2d(hidden_size * 2, hidden_size, 5, padding=2),
                nn.Conv2d(hidden_size, out_channels, 5, padding=2),
            ]
        )
        self.activation = nn.ReLU()
        self.down_scale = nn.MaxPool2d(2)
        self.up_scale = nn.ConvTranspose2d(
            hidden_size, hidden_size, 5, stride=2, padding=1
        )

    def forward(self, x):
        skip_connections = []

        for i, layer in enumerate(self.down_sampling_layers):
            x = self.activation(layer(x))
            if i < 2:
                skip_connections.append(x)
                x = self.down_scale(x)

        for j, layer in enumerate(self.up_sampling_layers):
            x = self.up_scale(x)
            if j > 0:
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
                x = self.activation(layer(x))
        return x

# %%