# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms as T, datasets as TD, utils as TU, models as TM
from torch.utils.data import DataLoader, Dataset

from torch_geometric.data import Data

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
from dataclasses import dataclass
import torchinfo
from einops import rearrange
from extentions.utils import log, funcs
from extentions.gymnasium_extentions import env as exenv
from extentions.string_ex import Japanese
import extentions.torch_extentions.nn as exnn
from extentions.utils.pathes import DATASET_PATH
from functools import lru_cache
from itertools import permutations


__device__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__file__ = str(Path(locals().get("__file__", "main.py")).resolve())
logger = log.initialize_logger(__file__)
logger.info(f"{__file__=}")
__root__ = Path(__file__).parent
print(f"{__device__=}\n{__root__=}\n{__file__=}")


# %%

transform = T.Compose(
    [
        T.ToTensor(),
    ]
)
target_transform = T.Compose(
    [
        T.Lambda(lambda x: [y.replace("\n", ". ").lower() for y in x]),
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
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True
)
valid_dataloader = DataLoader(
    dataset=valid_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True
)


# %%


class SentenceGraphEmbedding(nn.Module):
    def __init__(
        self,
        chars: str = Japanese.ENGLISH,
        embedding_dim: int = 384,
        window_size: int = 4,
        stride: int = 3,
    ):
        super().__init__()
        self.chars = chars
        self.chars_size = len(chars) + 1  # chars + EOS
        self.window_size = window_size
        self.stride = stride

        self.embedding = nn.Sequential(
            nn.Embedding(self.chars_size, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

    def get_recon_text(self, decoder_output):
        dist = torch.cdist(
            decoder_output, self.embedding(torch.arange(self.chars_size))
        )
        return "".join(
            [
                self.chars[i] if i != (len(self.chars) - 1) else ""
                for i in torch.argmin(dist, dim=1)
            ]
        )

    def to_graph(self, char_embeddings):
        x = torch.tensor(list(permutations(range(self.window_size), 2)))
        edges = set()
        for i in range(0, char_embeddings.shape[0] - self.window_size):
            for e in x + min(self.stride * i, len(char_embeddings) - self.window_size):
                edges.add(tuple(e))

        return Data(
            x=char_embeddings.T,
            edge_index=torch.tensor(list(sorted(edges))).T,
        )

    def get_char_embeddings(self, x: str) -> torch.Tensor:
        char_indexes = torch.tensor([self.chars.index(y) for y in x])
        if char_indexes.shape[0] < self.window_size * 2:
            char_indexes = F.pad(
                char_indexes,
                (0, self.window_size * 2 - len(char_indexes)),
                mode="constant",
                value=len(self.chars),
            )

        return self.embedding(char_indexes)

    def forward(self, x: str) -> torch.Tensor:
        emb = self.get_char_embeddings(x)
        graph = self.to_graph(emb)
        return emb

class ImageGraphEmbedding(nn.Module):
    ...


words = [
    "closeup of bins of food that include broccoli and bread.",
    "a meal is presented in brightly colored plastic trays.",
    "there are containers filled with different kinds of foods",
    "colorful dishes holding meat, vegetables, fruit, and bread.",
    "a bunch of trays that have different food.",
]


word = words[0]
cs = SentenceGraphEmbedding(embedding_dim=4)


# long_text = """
# Attention is the concentration of awareness on some phenomenon to the exclusion of other stimuli.[1] It is a process of selectively concentrating on a discrete aspect of information, whether considered subjective or objective. William James (1890) wrote that "Attention is the taking possession by the mind, in clear and vivid form, of one out of what seem several simultaneously possible objects or trains of thought. Focalization, concentration, of consciousness are of its essence."[2] Attention has also been described as the allocation of limited cognitive processing resources.[3] Attention is manifested by an attentional bottleneck, in terms of the amount of data the brain can process each second; for example, in human vision, only less than 1% of the visual input data (at around one megabyte per second) can enter the bottleneck,[4][5] leading to inattentional blindness.[6]

# Attention remains a crucial area of investigation within education, psychology, neuroscience, cognitive neuroscience, and neuropsychology. Areas of active investigation involve determining the source of the sensory cues and signals that generate attention, the effects of these sensory cues and signals on the tuning properties of sensory neurons, and the relationship between attention and other behavioral and cognitive processes, which may include working memory and psychological vigilance. A relatively new body of research, which expands upon earlier research within psychopathology, is investigating the diagnostic symptoms associated with traumatic brain injury and its effects on attention. Attention also varies across cultures.[7]

# The relationships between attention and consciousness are complex enough that they have warranted perennial philosophical exploration. Such exploration is both ancient and continually relevant, as it can have effects in fields ranging from mental health and the study of disorders of consciousness to artificial intelligence and its domains of research.

# Contemporary definition and research
# Prior to the founding of psychology as a scientific discipline, attention was studied in the field of philosophy. Thus, many of the discoveries in the field of attention were made by philosophers. Psychologist John B. Watson calls Juan Luis Vives the father of modern psychology because, in his book De Anima et Vita (The Soul and Life), he was the first to recognize the importance of empirical investigation.[8] In his work on memory, Vives found that the more closely one attends to stimuli, the better they will be retained.

# By the 1990s, psychologists began using positron emission tomography (PET) and later functional magnetic resonance imaging (fMRI) to image the brain while monitoring tasks involving attention. Considering this expensive equipment was generally only available in hospitals, psychologists sought cooperation with neurologists. Psychologist Michael Posner (then already renowned for his influential work on visual selective attention) and neurologist Marcus Raichle pioneered brain imaging studies of selective attention.[9] Their results soon sparked interest from the neuroscience community, which until then had simply been focused on monkey brains. With the development of these technological innovations, neuroscientists became interested in this type of research that combines sophisticated experimental paradigms from cognitive psychology with these new brain imaging techniques. Although the older technique of electroencephalography (EEG) had long been used to study the brain activity underlying selective attention by cognitive psychophysiologists, the ability of the newer techniques to actually measure precisely localized activity inside the brain generated renewed interest by a wider community of researchers. A growing body of such neuroimaging research has identified a frontoparietal attention network which appears to be responsible for control of attention.[10]
# """


emb = cs(word)
print(emb.shape)
# recon = cs.get_recon_text(emb)
# print(recon)z
# print(
#     emb.shape, len(long_text)
# )  # , sum(long_text[i] == recon[i] for i in range(len(long_text) - 1)) / len(long_text))

# # %%
# chars_emb = cs.get_char_embeddings(long_text)
# torch.concat([emb, cs.memory_embedding(emb.transpose(1, 0).unsqueeze(0)).squeeze(0).transpose(1, 0)], dim=1)
# print(cs(word).shape)

# %%
import networkx as nx
from torch_geometric.utils.convert import to_networkx

g = cs.to_graph(emb)
print(g)
nx.draw_networkx(
    to_networkx(g),
    labels={i: word[i] for i in range(len(word))},
    node_size=100,
    font_size=12,
    font_color="white",
)
