from pathlib import Path
from src.const import DATASET_PATH
import pandas as pd
import urllib.request
import json
from PIL import Image


class WITDataset:
    def __init__(self, wit_data_path: Path, with_image=True) -> None:
        self.wit_data_path = wit_data_path

        data_dir = DATASET_PATH.WIKIPEDIA_BASED_DATASET.value
        data_path_1 = tuple(data_dir.iterdir())[0]
        df = pd.read_csv(data_path_1, sep="\t", header=0)
        df = df[df["is_main_image"] == with_image]
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(urllib.request.urlopen(row["image_url"]))
        return row, image


class StairCaptionsDataset:
    def __init__(self, stair_caption_path: Path = Path("/mnt/d/Dataset/STAIR-captions")) -> None:
        self.path = stair_caption_path
        self.train = json.load((self.path / "stair_captions_v1.2_train.json").open())
        self.train_token = json.load((self.path / "stair_captions_v1.2_train_tokenized.json").open())
        self.val = json.load((self.path / "stair_captions_v1.2_val.json").open())
        self.val_token = json.load((self.path / "stair_captions_v1.2_val_tokenized.json").open())
