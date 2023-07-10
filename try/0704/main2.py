# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.utils.save_video import save_video

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torchvision
import torchvision.io as vio
import torchvision.transforms as transforms

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import GCNConv, Node2Vec
import torch_scatter
import cv2
import pygame
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation

from IPython.core.display import HTML

from tqdm import tqdm
from pathlib import Path
from itertools import count
from dataclasses import dataclass

import hydra
import string
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from functools import lru_cache
from gensim.models import FastText

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
from typing import NamedTuple
from pathlib import Path
from tqdm.auto import tqdm
from itertools import islice
from typing import Optional
import torchinfo

from extentions.utils import log, funcs
from extentions.gymnasium_extentions import env as exenv
from extentions.string_ex import Japanese
import extentions.torch_extentions.nn as exnn


logger = log.initialize_logger(__file__)
logger.info(f"{__file__=}")
__root__ = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE=}\n{__root__=}")

"""
行動空間
- テキスト出力

観測空間
- 出力したテキスト + 画像グラフ
"""


@lru_cache(maxsize=4096)
def levenshtein_distance(s, t):
    """文字列のレーベンシュタイン距離を計算する"""
    # 一方が空文字列なら、他方の長さが求める距離
    if not s:
        return len(t)
    if not t:
        return len(s)

    # 一文字目が一致なら、二文字目以降の距離が求める距離
    if s[0] == t[0]:
        return levenshtein_distance(s[1:], t[1:])

    # 一文字目が不一致なら、追加／削除／置換のそれぞれを実施し、
    # 残りの文字列についてのコストを計算する
    l1 = levenshtein_distance(s, t[1:])
    l2 = levenshtein_distance(s[1:], t)
    l3 = levenshtein_distance(s[1:], t[1:])

    # 追加／削除／置換を実施した分コスト（距離）1の消費は確定
    # 残りの文字列についてのコストの最小値を足せば距離となる
    return 1 + min(l1, l2, l3)


def tuplize(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return x


def to_tensor(x, dtype=torch.float32, device=DEVICE):
    if isinstance(x, list) and len(x) > 0 and not isinstance(x[0], torch.Tensor):
        x = np.array(x)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    return x.to(device)


def get_flat_params(module: nn.Module):
    return torch.cat([param.view(-1) for param in module.parameters()])


def set_params(module: nn.Module, new_flat_params):
    start_idx = 0
    for param in module.parameters():
        end_idx = int(start_idx + np.prod(list(param.shape)))
        param.data = torch.reshape(new_flat_params[start_idx:end_idx], param.shape)
        start_idx = end_idx


def get_flat_grads(f, net):
    flat_grads = torch.cat(
        [
            grad.view(-1)
            for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
        ]
    )
    return flat_grads


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10, success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params


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


def word2chars(word):
    if len(word) == 1:
        return word
    else:
        return word2chars(word[:-1]) + " " + word[:-1]


# %%


# caption_chars = []
# for img, captions in tqdm(train_dataset):
#     for caption in captions:
#         chars = word2chars(caption).split(" ")[1:]
#         chars.append(caption)
#         caption_chars.append(chars)

# print(len(caption_chars))
# %%
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# for image, label in train_dataloader:
#     print(image.shape)
#     exit()
# for (image_patches, edges_link, edges_attr), label in train_dataloader:
#     image_patches, edges_link, edges_attr = (
#         image_patches[0],
#         edges_link[0],
#         edges_attr[0],
#     )
#     print(image_patches.shape, image_patches.min(), image_patches.max())
#     print(edges_link.shape)
#     print(edges_attr.shape, edges_attr.min(), edges_attr.max())
#     x = GCNConv(27, 64)(image_patches.reshape(100, -1), edges_link, edges_attr)
#     x = nn.ReLU()(x)
#     x = GCNConv(64, 128)(x, edges_link, edges_attr)
#     x = nn.ReLU()(x)
#     x, _ = torch_scatter.scatter_max(x, torch.arange(100), dim=0)
#     x = nn.Linear(128, 100)(x)
#     x = nn.ReLU()(x)
#     x = nn.Linear(100, 1)(x)
#     x = image_patches * x.reshape(100, 1, 1, 1)
#     print(x.shape)
#     break

VALID_CHARS = Japanese.ENGLISH
print("Valid_Chars", VALID_CHARS)
# char_embedding_size = 4


class PolicyNetworkBase(nn.Module):
    # 現状から次の行動を決定する
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def get_distribution(self, states) -> torch.distributions.distribution.Distribution:
        raise NotImplementedError()

    def forward(
        self, states
    ) -> torch.distributions.distribution.Distribution:  # 状態をもとにアクションを決定
        states = to_tensor(states)
        return self.get_distribution(states)


class DiscreatePolicyNetwork(PolicyNetworkBase):
    # 離散的な行動をとる
    def get_distribution(self, states):
        probs = torch.softmax(self.net(states), dim=-1)
        return torch.distributions.Categorical(probs)


class ValueNetwork(nn.Module):
    # 状態から価値を予測する
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        states = to_tensor(states)
        return self.net(states)


class DiscreteDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.action_emb = nn.Embedding(action_dim, state_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_logits(self, states, actions):
        states = to_tensor(states)
        actions = to_tensor(actions)
        actions = self.action_emb(actions.long())
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))


class Expert(nn.Module):
    def __init__(self):
        super().__init__()

    def get_action(self, states):
        return VALID_CHARS.index(states["true"][len(states["text"])])


class ImageTextEnv(gym.Env):
    class Info(NamedTuple):
        target_image: torch.Tensor
        target_text: str

    def __init__(
        self,
        valid_chars=string.ascii_lowercase + string.digits + string.punctuation + " ",
        image_size=(256, 256),
        text_max_length=256,
        fontsize=20,
        fontname="ipamincho",
        capture_path: Optional[Path] = None,
    ):
        super(ImageTextEnv, self).__init__()
        self.valid_chars = valid_chars
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.max_word_length = 16
        # 0 is for end of sentence.
        self.action_space = gym.spaces.Discrete(len(self.valid_chars) + 1)
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=1, shape=(3, *image_size), dtype=np.float32
                ),
                "text": gym.spaces.Text(
                    max_length=text_max_length, charset=self.valid_chars
                ),
            }
        )
        self._action = self.valid_chars
        self._observation = {
            "image": torch.randn(3, *self.image_size),
            "text": "",
        }
        self._info = self.Info(
            target_image=torch.randn(3, *self.image_size), target_text=""
        )

        self.pygame = pygame
        self.pygame.init()
        self.fontsize = fontsize
        self.fontname = fontname
        self.screen = self.pygame.display.set_mode(
            (
                self.image_size[0] * 2 + self.fontsize * 4 + 128 * 3,
                self.image_size[1] + 32 + self.fontsize * 5,
            )
        )
        self.font = self.pygame.font.SysFont(self.fontname, self.fontsize)
        self.pygame.display.set_caption("MSCocoImageTextEnv")
        self.writer = None
        if capture_path is not None:
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            frame_rate = 120.0  # フレームレート
            fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # ファイル形式(ここではmp4)
            self.writer = cv2.VideoWriter(
                str(capture_path), fmt, frame_rate, self.screen.get_size()
            )

    @property
    def action(self):
        return self._action

    @property
    def observation(self):
        return self._observation

    @property
    def info(self):
        return self._info

    def reset(self, target_image, target_text):
        self._action = self.valid_chars
        self._observation = {
            "image": torch.rand(3, *self.image_size),
            "text": "",
        }
        self._info = self.Info(
            target_image=target_image,
            target_text=target_text,
        )
        return self.observation, self.info

    def step(self, action_index, next_image=None):
        if next_image is None:
            next_image = self.observation["image"]
        terminated, truncated, info = False, False, self.info
        reward = -levenshtein_distance(
            self.observation["text"], self.info.target_text
        ) / (len(self.info.target_text))

        if action_index == len(self.action):  # end of sentence
            truncated = True
            self.observation["text"] += "<EOS>"
        else:
            next_char = self.action[action_index]
            self.observation["text"] += next_char

        if len(self.observation["text"]) > self.text_max_length:
            terminated = True
            truncated = True

        self.observation["image"] = next_image

        # 文字を選択する→文字の埋め込み表現を入力として画像のノイズを除去するような学習を行っていく
        return self.observation, reward, terminated, truncated, info

    def pygame_quit(self):
        logger.info("Quit")
        if self.writer is not None and self.writer.isOpened():
            self.writer.release()
        self.pygame.quit()

    def render_wait(self, milliseconds=100):
        self.pygame.time.wait(milliseconds)

    def event_wait(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT or (
                event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE
            ):
                self.pygame_quit()

    def render_update(self):
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(
                cv2.cvtColor(
                    np.array(self.pygame.surfarray.array3d(self.screen).swapaxes(0, 1)),
                    cv2.COLOR_BGR2RGB,
                )
            )
        self.pygame.display.update()

    def render(self, *args, **kwargs):
        self.screen.fill((255, 255, 255))
        image_surf = self.pygame.surfarray.make_surface(
            (self.observation["image"].permute(2, 1, 0).numpy() * 255).astype(np.uint8)
        )
        true_image_surf = self.pygame.surfarray.make_surface(
            (self.info.target_image.permute(2, 1, 0).numpy() * 255).astype(np.uint8)
        )
        emb = kwargs.get("emb", torch.rand_like(self.observation["image"])).permute(
            1, 0
        )
        emb = emb.view(1, 1, emb.shape[0], emb.shape[1])
        emb = (
            torch.nn.functional.interpolate(emb, size=self.image_size, mode="bilinear")
            .cpu()
            .detach()
        )
        emb = emb.squeeze(0).squeeze(0)
        emb = (
            ((emb - emb.min()) / (emb.max() - emb.min()) * 255).numpy().astype(np.uint8)
        )
        emb_image_surf = self.pygame.surfarray.make_surface(emb)

        obs_text = self.observation["text"]
        text_surfs = [
            (
                self.font.render(obs_text, True, (0, 0, 0)),
                (32, 16 + self.image_size[1] + 32),
            ),
            (
                self.font.render(
                    f"loss={kwargs.get('loss', 0.0):.8f}", True, (0, 0, 0)
                ),
                (32, 16 + self.image_size[1] + 64),
            )
            # for i, text in enumerate(obs_text.split(" "))
        ]
        self.screen.blits(
            [
                (image_surf, (64, self.fontsize)),
                (
                    true_image_surf,
                    (64 + self.image_size[0] + self.fontsize * 2, self.fontsize),
                ),
                (
                    emb_image_surf,
                    (64 + self.image_size[0] * 2 + self.fontsize * 4, self.fontsize),
                ),
                *text_surfs,
            ]
        )

        self.event_wait()
        self.render_update()


class Text2Vec(nn.Module):
    def __init__(self, sentences: list[str] = [], char_vec_dim=256, device=DEVICE, model_path="fast_text.model"):
        super().__init__()
        if model_path:
            self.fast_text = FastText.load(model_path)
        else:
            self.fast_text = FastText(
                sentences, vector_size=char_vec_dim, window=8, min_count=1, workers=4
            )
            self.fast_text.train(sentences, total_examples=len(sentences), epochs=10000)
            self.fast_text.save("fast_text.model")
        self.device = device

    def forward(self, chars):
        return torch.from_numpy(self.fast_text.wv[chars].copy()).to(self.device)


# %%


# %%


# class Tmp:
#     def __iter__(self):
#         with open("caption_chars.txt", "r") as f:
#             for line in f:
#                 chars = line.strip().split(" ")
#                 if chars == [""]:
#                     continue
#                 else:
#                     yield [c for c in chars if c]


# tmp = list(iter(Tmp()))
# %%

from gensim.test.utils import get_tmpfile

# model = FastText(tmp, vector_size=512, window=5, min_count=1, workers=6)
# print(model.corpus_count)
# model.train(tmp, total_examples=model.corpus_count, epochs=100)
# model.save(get_tmpfile(str(__root__ / "fast_text.model")))
model = FastText.load(get_tmpfile(str(__root__ / "fast_text.model")))


# %%


class Text2Image(nn.Module):
    def __init__(
        self,
        valid_chars,
        char_emb_size=(16, 16),
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
        return context_vec


class GAIL(nn.Module):
    def __init__(
        self,
        state_dim=128,
        action_dim=len(VALID_CHARS),
        char_vec_dim=32,
        policy_hidden_dim=256,
        value_hidden_dim=256,
    ):
        super().__init__()
        self.state_dim = state_dim + char_vec_dim
        print("STATE DIM", self.state_dim, "ACTION DIM", action_dim)
        self.action_dim = action_dim

        self.policy_net = DiscreatePolicyNetwork(
            self.state_dim, self.action_dim, hidden_dim=policy_hidden_dim
        )
        self.value_net = ValueNetwork(self.state_dim, hidden_dim=value_hidden_dim)
        self.discriminator = DiscreteDiscriminator(
            self.state_dim, self.action_dim, hidden_dim=value_hidden_dim
        )
        self.char_emb = Text2Vec(model_path="fast_text.model")

    def get_text_emb(self, text):
        return self.char_emb(text)

    def get_action(self, states):
        inputs = torch.cat([states["emb"], self.get_text_emb(states["text"])])
        with torch.inference_mode():
            dist = self.policy_net(inputs)  # type: ignore
        return dist.sample().detach().cpu().numpy()

    def train(self, env: ImageTextEnv, agent: Expert):
        num_steps_per_epochs = 10000
        num_epochs = 1000
        gae_gamma = 0.99
        gae_lambda = 0.99
        max_kl = 0.025
        epsilon = 0.01
        normarize_advantage = True
        lambda_ = 0.001
        cg_damping = 0.1

        agent_observations = []
        agent_actions = []
        agent_rewards = []
        max_reward = -np.inf
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters())

        step = 0
        while step < num_steps_per_epochs:
            observation, _ = env.reset()
            agent_rewards_step = []
            agent_observations_step = []
            terminated, truncated = False, False
            reward = -np.inf

            while (
                (not terminated) and (not truncated) and (step < num_steps_per_epochs)
            ):
                action = agent.get_action(observation)
                obs = torch.cat(
                    [observation["emb"], self.get_text_emb(observation["text"])]
                )
                agent_observations_step.append(obs)
                agent_observations.append(obs)
                agent_actions.append(action)

                observation, reward, terminated, truncated, info = env.step(action)
                agent_rewards_step.append(reward)

                step += 1

                if step % 1000 == 0:
                    print(
                        f"step: {step}, label: {observation['true']}, text: {observation['text']}, reward: {reward}"
                    )

            agent_rewards.append(torch.sum(to_tensor(agent_rewards_step)))
            agent_observations_step = torch.vstack(agent_observations_step)
            agent_rewards_step = to_tensor(agent_rewards_step)

        agent_rewards_mean = to_tensor(agent_rewards).mean()
        # max_reward = max(max_reward, agent_rewards_mean.item())
        print(f"agent_rewards_mean: {agent_rewards_mean}")

        agent_observations = torch.vstack(agent_observations)
        agent_actions = to_tensor(agent_actions, dtype=torch.long)
        reward_epoch_means = []

        for i in range(num_epochs):
            rewards_epochs = []
            observations_epochs = []
            actions_epochs = []
            returns_epochs = []
            advantages_epochs = []
            gae_gammas_epochs = []

            steps = 0
            while steps < num_steps_per_epochs:
                rewards_step = []
                observations_step = []
                actions_step = []
                gae_gammas_step = []
                lambdas_step = []
                t = 0
                terminated, truncated = False, False

                observation, _ = env.reset()

                while (
                    (not terminated)
                    and (not truncated)
                    and (steps < num_steps_per_epochs)
                ):
                    action = self.get_action(observation)
                    observation, reward, terminated, truncated, info = env.step(action)

                    obs = torch.cat(
                        [observation["emb"], self.get_text_emb(observation["text"])]
                    )
                    observations_step.append(obs)
                    observations_epochs.append(obs)

                    actions_step.append(action)
                    actions_epochs.append(action)

                    rewards_step.append(reward)
                    gae_gammas_step.append(gae_gamma**t)
                    lambdas_step.append(gae_lambda**t)

                    t += 1
                    steps += 1

                    if steps % 1000 == 0:
                        print(
                            f"step: {steps}, observation_text: {observation['text']}, label: {observation['true']}, reward: {reward}"
                        )
                rewards_epochs.append(torch.sum(to_tensor(rewards_step)))
                rewards_step = to_tensor(rewards_step)
                observations_step = torch.vstack(observations_step)
                actions_step = to_tensor(actions_step, dtype=torch.long)
                gae_gammas_step = to_tensor(gae_gammas_step)
                lambdas_step = to_tensor(lambdas_step)
                costs_step = (
                    -torch.log(self.discriminator(observations_step, actions_step))
                    .squeeze()
                    .detach()
                )

                discriminator_costs_step = gae_gammas_step * costs_step
                discriminator_returns_step = to_tensor(
                    [discriminator_costs_step[i:].sum() for i in range(t)]
                )
                returns_step = discriminator_returns_step / gae_gammas_step
                returns_epochs.append(returns_step)

                self.value_net.eval()
                current_values = self.value_net(observations_step).detach()
                next_values = torch.cat(
                    (self.value_net(observations_step)[1:], to_tensor([[0.0]]))
                ).detach()
                deltas_step = (
                    costs_step.unsqueeze(-1) + gae_gamma * next_values - current_values
                )
                advantages_step = to_tensor(
                    [
                        (
                            (gae_gammas_step * lambdas_step)[: t - j].unsqueeze(-1)
                            * deltas_step[j:]
                        ).sum()
                        for j in range(t)
                    ]
                )
                advantages_epochs.append(advantages_step)
                gae_gammas_epochs.append(gae_gammas_step)

            reward_epoch_means.append(to_tensor(rewards_epochs).mean())

            observations_epochs = torch.vstack(observations_epochs)
            actions_epochs = to_tensor(actions_epochs, dtype=torch.long)
            returns_epochs = torch.cat(returns_epochs)
            advantages_epochs = torch.cat(advantages_epochs)
            gae_gammas_epochs = torch.cat(gae_gammas_epochs)

            if normarize_advantage:
                advantages_epochs = (
                    advantages_epochs - advantages_epochs.mean()
                ) / advantages_epochs.std()

            self.discriminator.train()
            novelty_scores = self.discriminator.get_logits(
                observations_epochs, actions_epochs
            )
            agent_scores = self.discriminator.get_logits(
                agent_observations, agent_actions
            )

            self.discriminator_optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                agent_scores, torch.zeros_like(agent_scores)
            ) + torch.nn.functional.binary_cross_entropy_with_logits(
                novelty_scores, torch.ones_like(novelty_scores)
            )
            loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            print(
                f"Epoch {i + 1},  Reward Mean: {reward_epoch_means[-1]}, Loss: {loss.item():.4f}"
            )

            if reward_epoch_means[-1] > max_reward:
                print(f"- Saving model with reward {reward_epoch_means[-1]}")
                torch.save(
                    self.policy_net.state_dict(),
                    Path(__file__).parent / "policy_net.pt",
                )
                max_reward = reward_epoch_means[-1]

            self.value_net.train()
            pred_values_params = get_flat_params(self.value_net).detach()
            prev_values_epochs = self.value_net(observations_epochs).detach()
            grad_diffs = get_flat_grads(
                (prev_values_epochs - self.value_net(observations_epochs))
                .pow(2)
                .mean(),
                self.value_net,
            )
            g = get_flat_grads(
                (
                    (-1)
                    * (
                        self.value_net(observations_epochs).squeeze() - returns_epochs
                    ).pow(2)
                ).mean(),
                self.value_net,
            )

            def Hv(v):
                return get_flat_grads(torch.dot(grad_diffs, v), self.value_net).detach()

            s = conjugate_gradient(Hv, g)
            Hs = Hv(s)
            alpha = torch.sqrt(2 * epsilon / torch.dot(s, Hs))

            new_params = pred_values_params + alpha * s
            set_params(self.value_net, new_params)

            self.policy_net.train()
            prev_policy_params = get_flat_params(self.policy_net).detach()
            prev_distribution = self.policy_net(observations_epochs)

            def L():
                distb = self.policy_net(observations_epochs)
                return (
                    advantages_epochs
                    * torch.exp(
                        distb.log_prob(actions_epochs)
                        - prev_distribution.log_prob(actions_epochs).detach()
                    )
                ).mean()

            def kld():
                distb = self.policy_net(observations_epochs)
                old_probs = prev_distribution.probs.detach()
                probs = distb.probs
                return (
                    (old_probs * (torch.log(old_probs) - torch.log(probs)))
                    .sum(-1)
                    .mean()
                )

            grad_kld_old_param = get_flat_grads(kld(), self.policy_net)

            def Hv(v):
                return (
                    get_flat_grads(
                        torch.dot(grad_kld_old_param, v), self.policy_net
                    ).detach()
                    + cg_damping * v
                )

            g = get_flat_grads(L(), self.policy_net).detach()
            s = conjugate_gradient(Hv, g)
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g,
                s,
                Hs,
                max_kl,
                L,
                kld,
                prev_policy_params,
                self.policy_net,
                max_iter=100,
                success_ratio=0.05,
            )

            disc_causal_entropy = (
                (-1)
                * gae_gammas_epochs
                * self.policy_net(observations_epochs).log_prob(actions_epochs)
            ).mean()

            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.policy_net
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.policy_net, new_params)
        return agent_rewards_mean, reward_epoch_means


# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
from itertools import cycle

env = ImageTextEnv(
    cycle(train_dataset), image_embedding_size=3072
)
agent = Expert()

gail = GAIL(
    state_dim=env.image_embedding_size,
    char_vec_dim=256,
    policy_hidden_dim=1024,
    value_hidden_dim=1024,
)
agent.to(DEVICE)
gail.to(DEVICE)
gail.train(env, agent)
