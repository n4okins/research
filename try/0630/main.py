# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T, datasets as TD, utils as TU, models as TM
from torch.utils.data import DataLoader
from src.const import DATASET_PATH
from PIL import Image
from matplotlib import pyplot as plt
import pygame
from pprint import pprint
import numpy as np
import string
import gymnasium as gym
from collections import namedtuple
from pathlib import Path
from tqdm.auto import tqdm
from itertools import islice
from typing import Optional
import torchinfo

from extentions.utils import log, funcs
from extentions.gymnasium_extentions import env as exenv
from extentions.string_ex import Japanese
import extentions.torch_extentions.nn as exnn
from extentions.gymnasium_extentions.models import (
    GAILActor,
    GAILActorCritic,
    GAILCritic,
    GAILDiscriminator,
)


logger = log.initialize_logger(__file__)
logger.info(f"{__file__=}")
__root__ = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE=}\n{__root__=}")
# %%

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
    ]
)
target_transform = T.Compose(
    [
        T.Lambda(lambda x: [y.replace("\n", ". ").lower() for y in x]),
        # T.Lambda(lambda x: ""),
    ]
)
train_dataset = TD.CocoCaptions(
    root=str(DATASET_PATH.COCO_TRAIN2014_DATASET),
    annFile=str(DATASET_PATH.COCO_TRAIN2014_CAPTIONS_JSON),
    transform=transform,
    target_transform=target_transform,
)
valid_dataset = TD.CocoCaptions(
    root=str(DATASET_PATH.COCO_VAL2014_DATASET),
    annFile=str(DATASET_PATH.COCO_VAL2014_CAPTIONS_JA_JSON),
    transform=transform,
    target_transform=target_transform,
)


def sample_plot(idx=np.random.randint(0, len(train_dataset))):
    img, labels = train_dataset[idx]
    label = labels[np.random.randint(0, len(labels))]
    print(f"{idx=} {img.shape=} {label=}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img.permute(1, 2, 0))
    axes[0].set_title(label)
    axes[0].axis("off")

    axes[1].imshow(torch.rand_like(img).permute(1, 2, 0))
    axes[1].set_title(label)
    axes[1].axis("off")

    plt.show()


# sample_plot()

# %%


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_action_index(self, states) -> int:
        action_index = self.env.action_space.sample()
        return action_index


class Expert(Agent):
    def __init__(self, env: gym.Env, valid_chars: str) -> None:
        super().__init__(env)
        self.valid_chars = valid_chars

    def get_action_index(self, states, true_text: str) -> int:
        index = len(states["text"])
        if index < len(true_text):
            # logger.info(f"{index=} {true_text[index]=}")
            return self.valid_chars.index(true_text[index])
        else:
            return len(self.valid_chars)  # end of sentence


class Text2Vec(nn.Module):
    def __init__(
        self, valid_chars, char_emb_size=(32, 32), device=DEVICE, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.emb_size = char_emb_size
        self.valid_chars = valid_chars
        self.device = device
        self.char_embedding = nn.Embedding(
            len(valid_chars) + 1, char_emb_size[0] * char_emb_size[1], device=device,
            _freeze=True
        )

    def forward(
        self, text: str, context_vec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if context_vec is None:
            context_vec = torch.eye(max(self.emb_size), device=self.device)
        context_vec = context_vec.clone()
        for t in text:
            context_vec @= self.char_embedding(
                torch.tensor(self.valid_chars.index(t), device=self.device)
            ).view(*self.emb_size)
            context_vec = torch.cos(context_vec)
        return context_vec


chars = Japanese.ENGLISH


class Text2Image(nn.Module):
    def __init__(
        self,
        valid_chars,
        char_emb_size=(256, 256),
        img_size=(256, 256),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.char_emb_size = char_emb_size
        self.img_size = img_size
        self.device = DEVICE
        scale_count = 1

        def create_convt2d(in_channels, out_channels):
            nonlocal scale_count
            layer = nn.Sequential(
                nn.LayerNorm((in_channels, char_emb_size[0] * scale_count, char_emb_size[1] * scale_count)),  # type: ignore
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.SiLU(),
                nn.ConvTranspose2d(out_channels, out_channels, 1),
                nn.SiLU(),
            )
            scale_count *= 2
            return layer

        scale = torch.log2(
            torch.as_tensor(min(self.img_size) // min(self.char_emb_size))
        )
        self.text2vec = Text2Vec(valid_chars, char_emb_size, device=self.device)
        self.vec2image = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ConvTranspose2d(64, 128, 1),
            nn.SiLU(),
            *[create_convt2d(128, 128) for _ in range(int(scale) - 1)],
            create_convt2d(128, 64),
            nn.ConvTranspose2d(64, 3, 1),
            nn.Sigmoid()
            # nn.LayerNorm((4, char_emb_size[0], char_emb_size[1])), nn.SiLU(),
        ).to(self.device)

    def forward(
        self, text: str, context_vec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        context_vec = self.text2vec(text, context_vec).view(1, 1, *self.char_emb_size)
        context_vec = self.vec2image(context_vec)
        return context_vec  # type: ignore


# %%
batch_size = 32
num_label_epoch = 1
batches = torch.randperm(len(train_dataset)).split(batch_size)
text2img = Text2Image(chars, char_emb_size=(32, 32), img_size=(256, 256))
optimizer = torch.optim.Adam(text2img.parameters(), lr=1e-3)
text2vec_optimizer = torch.optim.Adam(text2img.text2vec.parameters(), lr=1e-3)
vocab: set[str] = set()
print(text2img)
print(torchinfo.summary(text2img.vec2image, (1, 1, 32, 32)))

if (__root__ / "text2img.pt").exists():
    state = torch.load(__root__ / "text2img.pt")
    text2img.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    vocab = state["vocab"]

# %%


def plot_test(save_to):
    texts = [
        ["blue", "red", "green", "yellow"],
        ["child", "adult", "old", "young"],
        ["some", "group", "one", "many"],
        ["soccer", "tennis", "basketball", "baseball"],
    ]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(
                text2img(texts[i][j])
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            axes[i, j].axis("off")
            axes[i, j].set_title(texts[i][j])
    plt.savefig(save_to)


if (__root__ / "text2img.pt").exists():
    text = "a page from a book is shown with an illustration of "
    print(text2img.text2vec.char_embedding.weight.data)
    vec = text2img.text2vec(text)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    img = text2img.vec2image(vec.view(1, 1, *text2img.char_emb_size))
    ax[0].imshow(img.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    ax[1].imshow(vec.detach().cpu().numpy())
    ax[0].set_title("image")
    ax[1].set_title("vector")
    ax[0].axis("off")
    ax[1].axis("off")
    plt.suptitle(text)
    plt.show()
    plot_test(__root__ / "text2img_prev_model.png")
# %%

capture_path = __root__ / "capture" / "07042.mp4"
logger.info(f"{capture_path=}")

env = exenv.ImageTextEnv(
    valid_chars=chars, image_size=(256, 256), capture_path=capture_path
)
expert = Expert(env, env.valid_chars)
criterion = nn.KLDivLoss(reduction="batchmean")

text2img.to(DEVICE)
text2img.train()

for i, batch in enumerate(tqdm(batches)):  # type: ignore
    for j in tqdm(range(len(batch)), leave=False):
        torch.save(
            {
                "model": text2img.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vocab": vocab,
            },
            __root__ / "text2img.pt",
        )
        img, labels = train_dataset[batch[j]]
        for x, label in enumerate(labels):
            text2vec_optimizer.zero_grad()
            loss = torch.tensor(0.0, device=DEVICE)
            for y, label in enumerate(labels):
                if x == y:
                    continue

                loss += F.mse_loss(
                    text2img.text2vec(labels[x]), text2img.text2vec(labels[y])
                )
            loss.backward()
            text2vec_optimizer.step()

        for t, label in enumerate(tqdm(labels, leave=False)):
            for s in range(num_label_epoch):
                obs, _ = env.reset(img, label)
                while True:
                    loss = torch.tensor(0.0, device=DEVICE)
                    optimizer.zero_grad()

                    compliete_image = text2img(env.info.target_text).squeeze(0)
                    next_image = text2img(obs["text"]).squeeze(0)

                    loss += F.mse_loss(next_image.cpu(), env.info.target_image.cpu())
                    loss += torch.exp(
                        -text2img.text2vec.char_embedding.weight.data.cpu().std()
                    )
                    action = expert.get_action_index(obs, env.info.target_text)
                    obs, reward, terminated, truncated, info = env.step(
                        action, next_image.cpu().detach()
                    )
                    loss.backward()
                    optimizer.step()

                    if terminated or truncated:
                        tqdm.write(
                            f"{i=:06d} {j=} {t=} {s=} {loss.item()=:.8f} | {env.observation['text']} \n {text2img.text2vec.char_embedding.weight.data.cpu().std()=:.8f}"
                        )
                        break

                    vocab.add(env.observation["text"][-1])
                    env.render(
                        loss=loss.item(),
                        emb=text2img.text2vec(env.observation["text"]).detach(),
                    )
                    env.render_wait(1)

    plot_test(__root__ / f"text2img_{i:04d}.png")

env.pygame_quit()

# %%
# epochs = 10
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# save_dir = __root__ / "results" / "autoencoder"
# save_fig_dir = save_dir / "fig"
# save_fig_dir.mkdir(parents=True, exist_ok=True)

# for e in tqdm(range(epochs), position=0):
#     loss = torch.tensor(0.0, device=DEVICE)

#     for image, labels in tqdm(train_dataloader, position=1, leave=False):
#         image, labels = image.to(DEVICE), labels
#         pred = model(image)
#         optimizer.zero_grad()
#         loss += F.mse_loss(pred, image)
#         loss.backward()
#         optimizer.step()

#     torch.save(
#         {
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": e,
#             "loss": loss.item(),
#         },
#         save_dir / f"model_{e}.pth",
#     )
#     plt.cla()
#     plt.clf()
#     plt.close()
#     fig, axes = plt.subplots(10, 3, figsize=(10, 4 * 8))
#     for i, (image, labels) in enumerate(
#         tqdm(islice(valid_dataloader, 10), position=1, leave=False)
#     ):
#         image, labels = image.to(DEVICE), labels
#         emb = model.enc(image)
#         pred = model.dec(image)
#         emb = emb.detach().cpu().permute(0, 2, 3, 1).numpy()  # type: ignore
#         image, pred = image.detach().cpu().permute(0, 2, 3, 1).numpy(), pred.detach().cpu().permute(0, 2, 3, 1).numpy()  # type: ignore
#         image, emb, pred = image[0], emb[0], pred[0]
#         ax = axes[i]
#         ax[0].imshow(image)  # type: ignore
#         ax[0].set_title("image")
#         ax[1].imshow(emb)  # type: ignore
#         ax[1].set_title("emb")
#         ax[2].imshow(pred)  # type: ignore
#         ax[2].set_title("pred")
#         for a in ax:
#             a.axis("off")
#     plt.savefig(save_fig_dir / "ViT.png")
#     break

# %%
# for i in torch.randperm(len(train_dataset)):  # type: ignore
# for j, label in enumerate(labels):
#     obs, _ = env.reset(img, label)

#     while True:
#         action = expert.get_action_index(obs, env.info.target_text)
#         obs, reward, terminated, truncated, info = env.step(action)

#         if terminated or truncated:
#             print(f"{i=:06d} {j=} | {env.observation['text']}")
#             break

#         env.render()
#         env.render_wait(10)
#     env.render_wait(100)
# if i >= 0:
#     break

# env.pygame_quit()
# %%
