# https://engineers.ntt.com/entry/2022/12/20/084912
# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from pathlib import Path
from extentions.utils import log


__device__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__file__ = str(Path(locals().get("__file__", "main.py")).resolve())
logger = log.initialize_logger(__file__)
logger.info(f"{__file__=}")
__root__ = Path(__file__).parent
print(f"{__device__=}\n{__root__=}\n{__file__=}")

# %%
# データの読み込み（pandas）
# https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
df_recipes = pd.read_csv(__root__ / "data/recipes.csv")
df_reviews = pd.read_csv(__root__ / "data/reviews.csv")
# データ準備
df_reviews = df_reviews[
    df_reviews.RecipeId.isin(df_recipes["RecipeId"].unique())
]  # 不要データ除外
df_recipes["RecipeServings"] = df_recipes["RecipeServings"].fillna(
    df_recipes["RecipeServings"].median()
)  # 欠損値補完

# %%
# ユーザノードとレシピノードのIDマップ作成
unique_user_id = df_reviews["AuthorId"].unique()
unique_user_id = pd.DataFrame(
    data={
        "user_id": unique_user_id,
        "mappedID": pd.RangeIndex(len(unique_user_id)),
    }
)
unique_recipe_id = df_reviews["RecipeId"].unique()
unique_recipe_id = pd.DataFrame(
    data={
        "recipe_id": unique_recipe_id,
        "mappedID": pd.RangeIndex(len(unique_recipe_id)),
    }
)
review_user_id = pd.merge(
    df_reviews["AuthorId"],
    unique_user_id,
    left_on="AuthorId",
    right_on="user_id",
    how="left",
)
review_recipe_id = pd.merge(
    df_reviews["RecipeId"],
    unique_recipe_id,
    left_on="RecipeId",
    right_on="recipe_id",
    how="left",
)

# %%
# ユーザIDとレシピIDのエッジ情報をTensorへ変換
tensor_review_user_id = torch.from_numpy(review_user_id["mappedID"].values)
tensor_review_recipe_id = torch.from_numpy(review_recipe_id["mappedID"].values)
tensor_edge_index_user_to_recipe = torch.stack(
    [tensor_review_user_id, tensor_review_recipe_id],
    dim=0,
)
print(
    f"{tensor_review_user_id=}\n{tensor_review_recipe_id=}\n{tensor_edge_index_user_to_recipe=}"
)
# %%
# レシピノードの特徴量定義
recipe_feature_cols = [
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
    "RecipeServings",
]
df_recipes_feature = pd.merge(
    df_recipes, unique_recipe_id, left_on="RecipeId", right_on="recipe_id", how="left"
)
df_recipes_feature = df_recipes_feature.sort_values("mappedID").set_index("mappedID")
df_recipes_feature = df_recipes_feature[df_recipes_feature.index.notnull()]
df_recipes_feature = df_recipes_feature[recipe_feature_cols]

# 標準化
scaler = StandardScaler()
scaler.fit(df_recipes_feature)
scaler.transform(df_recipes_feature)
df_recipes_feature = pd.DataFrame(
    scaler.transform(df_recipes_feature), columns=df_recipes_feature.columns
)

# レシピノードの特徴量をTensorへ変換
tensor_recipes_feature = torch.from_numpy(df_recipes_feature.values).to(torch.float)


# %%
# HeteroDataオブジェクトの作成
data = HeteroData()
data["user"].node_id = torch.arange(len(unique_user_id))
data["recipe"].node_id = torch.arange(len(unique_recipe_id))
data["recipe"].x = tensor_recipes_feature
data["user", "review", "recipe"].edge_index = tensor_edge_index_user_to_recipe
data = T.ToUndirected()(data)

print(
    f"{tensor_review_user_id.shape=}\n{torch.arange(len(unique_user_id)).shape=}"
    + f"\n{tensor_review_recipe_id.shape=}\n{torch.arange(len(unique_recipe_id)).shape=}"
    + f"\n{tensor_edge_index_user_to_recipe.shape=}\n\n\n"
)
print(f"{data=}")
# %%
# 学習・評価用のデータ分割
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=1,
    add_negative_train_samples=False,
    edge_types=("user", "review", "recipe"),
    rev_edge_types=("recipe", "rev_review", "user"),
)
train_data, val_data, test_data = transform(data)  # type: ignore
print(train_data, val_data, test_data)

# %%
# 学習用データローダー定義
edge_label_index = train_data["user", "review", "recipe"].edge_label_index
edge_label = train_data["user", "review", "recipe"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=1,
    edge_label_index=(("user", "review", "recipe"), edge_label_index),
    edge_label=edge_label,
    batch_size=256,
    shuffle=True,
)

# 検証用データローダー定義
edge_label_index = val_data["user", "review", "recipe"].edge_label_index
edge_label = val_data["user", "review", "recipe"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "review", "recipe"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 256,
    shuffle=False,
)


# %%


class GNN(Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Classifier(Module):
    def forward(
        self, x_user: Tensor, x_recipe: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_recipe = x_recipe[edge_label_index[1]]

        return (edge_feat_user * edge_feat_recipe).sum(dim=-1)


class Model(Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.recipe_lin = torch.nn.Linear(10, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.recipe_emb = torch.nn.Embedding(data["recipe"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "recipe": self.recipe_lin(data["recipe"].x)
            + self.recipe_emb(data["recipe"].node_id),
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["user"],
            x_dict["recipe"],
            data["user", "review", "recipe"].edge_label_index,
        )

        return pred


# %%
def train(model, loader, device, optimizer, epoch):
    model.train()
    for epoch in range(1, epoch):
        total_loss = total_samples = 0
        for batch_data in tqdm(loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            pred = model(batch_data)
            loss = F.binary_cross_entropy_with_logits(
                pred, batch_data["user", "review", "recipe"].edge_label
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_samples += pred.numel()
        print(f"Epoch: {epoch:04d}, Loss: {total_loss / total_samples:.4f}")


def validation(model, loader, device, optimizer):
    y_preds = []
    y_trues = []
    model.eval()
    for batch_data in tqdm(loader):
        with torch.no_grad():
            batch_data = batch_data.to(device)
            pred = model(batch_data)
            y_preds.append(pred)
            y_trues.append(batch_data["user", "review", "recipe"].edge_label)

    y_pred = torch.cat(y_preds, dim=0).cpu().numpy()
    y_true = torch.cat(y_trues, dim=0).cpu().numpy()
    auc = roc_auc_score(y_true, y_pred)
    return auc, y_pred, y_true


# パラメータセット
model = Model(hidden_channels=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)

# 学習・評価
train(model, train_loader, device, optimizer, 6)
auc, y_pred, y_true = validation(model, val_loader, device, optimizer)

# 精度確認（ROC-AUC曲線）
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label=f"AUC: {auc:.3f}")
plt.xlabel("FPR: False positive rate")
plt.ylabel("TPR: True positive rate")
plt.legend(loc="lower right")
plt.grid()

# %%
