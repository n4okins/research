# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.datasets as TD

import torch_geometric.nn as TgNN
import torch_geometric.transforms as TgT
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader


import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict


from extentions.utils import log
from extentions.utils.pathes import DATASET_PATH
from extentions.string_ex import Japanese


__device__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__file__ = str(Path(locals().get("__file__", "main.py")).resolve())
logger = log.initialize_logger(__file__)
logger.info(f"{__file__=}")
__root__ = Path(__file__).parent
print(f"{__device__=}\n{__root__=}\n{__file__=}")


# %%
class DatasetWithIndex:
    def __init__(self, dataset, image_transform=None, to_graph_transform=None):
        self.dataset = dataset
        self.image_transform = image_transform or (lambda x: x)
        self.to_graph_transform = to_graph_transform or (lambda x: x)

    def __getitem__(self, index):
        _img, label = self.dataset[index]
        img = self.image_transform(_img)
        graph = self.to_graph_transform(_img)
        return index, img, graph, label

    def __len__(self):
        return len(self.dataset)


# %%
def plot(batches, idx=0):
    index, img, graph, labels = batches
    index, img, graph, labels = index[idx], img[idx], graph[idx], labels[idx]
    pos = {
        i: (graph.pos[i, 0], graph.pos[:, 1].max() - graph.pos[i, 1])
        for i in range(graph.num_nodes)
    }
    node_color = graph.x
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    ax[0].imshow(img.permute(1, 2, 0).detach().cpu())
    ax[0].axis("off")

    logger.info(f"Draw graph start ... | {graph=}")

    nx.draw_networkx(
        to_networkx(graph),
        pos=pos,
        node_color=node_color,
        node_size=36,
        with_labels=False,
        arrowsize=3,
        ax=ax[1],
    )
    ax[1].axis("off")
    fig.set_tight_layout(True)
    fig.suptitle(labels[0])
    plt.show()

# %%
# image_transform = T.Compose(
#     [
#         T.ToTensor(),
#         T.Resize(256, antialias=True),
#         T.CenterCrop(256),
#     ]
# )
# text_transform = T.Compose(
#     [
#         T.Lambda(lambda x: [y.replace("\n", ". ").lower() for y in x]),
#     ]
# )
# img2graph_transform = T.Compose(
#     [
#         T.ToTensor(),
#         TgT.ToSLIC(n_segments=4096),
#         TgT.KNNGraph(k=8),
#     ]
# )

# train_dataset = TD.CocoCaptions(
#     root=str(DATASET_PATH.COCO_TRAIN2014_DATASET),
#     annFile=str(DATASET_PATH.COCO_TRAIN2014_CAPTIONS_JSON),
#     target_transform=text_transform,
# )
# valid_dataset = TD.CocoCaptions(
#     root=str(DATASET_PATH.COCO_VAL2014_DATASET),
#     annFile=str(DATASET_PATH.COCO_VAL2014_CAPTIONS_JSON),
#     target_transform=text_transform,
# )
# train_dataset = DatasetWithIndex(
#     train_dataset,
#     image_transform=image_transform,
#     to_graph_transform=img2graph_transform,
# )
# valid_dataset = DatasetWithIndex(
#     valid_dataset,
#     image_transform=image_transform,
#     to_graph_transform=img2graph_transform,
# )


# # %%
# train_dataloader = DataLoader(
#     dataset=train_dataset, batch_size=256, shuffle=True, num_workers=6
# )
# valid_dataloader = DataLoader(
#     dataset=valid_dataset, batch_size=256, shuffle=True, num_workers=6
# )


# %%time
valid_chars = Japanese.ENGLISH

# cnt = 0
# for indexes, imgs, graphs, labels in tqdm(train_dataloader):
#     data = HeteroData()
#     data["images"].id = indexes
#     data["images"].image = imgs
#     data["images"].graph = graphs
#     data["images"].labels = labels
#     data["images"].captions = defaultdict(list)
#     for label in zip(labels):
#         for la in label:
#             for i, chrs in enumerate(la):
#                 caption = []
#                 for c in chrs:
#                     try:
#                         caption.append(valid_chars.index(c))
#                     except:
#                         print(chrs)
#                         exit()
#                 data["images"].captions[i].append(torch.tensor(caption))

#     data["chars"].id = torch.arange(len(valid_chars) + 1)  # +1 for EOS

#     edges = list()
#     for i, image_id in enumerate(data["images"].id):
#         for j, caption in enumerate(data["images"].captions[i]):
#             for t, c in enumerate(caption):
#                 edges.append((i, c))

#     edges = torch.tensor(list(edges)).contiguous().t()  # type: ignore
#     data["images", "captions", "chars"].edge_index = edges
#     # heteros.append(data)
#     torch.save(data, f"/mnt/d/Dataset/mscoco/graph/train/hetero_{cnt:04d}.pt")
#     cnt += 1
# # print("".join([valid_chars[c.item()] for c in data["images"].captions[0][4]]))

# %%
# cnt = 0
# for indexes, imgs, graphs, labels in tqdm(valid_dataloader):
#     data = HeteroData()
#     data["images"].id = indexes
#     data["images"].image = imgs
#     data["images"].graph = graphs
#     data["images"].labels = labels
#     data["images"].captions = defaultdict(list)
#     for label in zip(labels):
#         for la in label:
#             for i, chrs in enumerate(la):
#                 caption = []
#                 for c in chrs:
#                     try:
#                         caption.append(valid_chars.index(c))
#                     except:
#                         print(chrs)
#                         exit()
#                 data["images"].captions[i].append(torch.tensor(caption))

#     data["chars"].id = torch.arange(len(valid_chars) + 1)  # +1 for EOS

#     edges = list()
#     for i, image_id in enumerate(data["images"].id):
#         for j, caption in enumerate(data["images"].captions[i]):
#             for t, c in enumerate(caption):
#                 edges.append((i, c))

#     edges = torch.tensor(list(edges)).contiguous().t()  # type: ignore
#     data["images", "captions", "chars"].edge_index = edges
#     # heteros.append(data)
#     torch.save(data, f"/mnt/d/Dataset/mscoco/graph/val/hetero_{cnt:04d}.pt")
#     cnt += 1

# %%


class GNN(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.conv1 = TgNN.SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = TgNN.SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = TgNN.SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        return x


# %%
