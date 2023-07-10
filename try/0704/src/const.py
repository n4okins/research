from types import SimpleNamespace
from pathlib import Path


class DATASET_PATH(SimpleNamespace):
    WIKIPEDIA_BASED_DATASET: Path = Path("/mnt/d/Dataset/wit/japanese")

    COCO_TRAIN2014_DATASET: Path = Path("/mnt/d/Dataset/mscoco/train2014")
    COCO_TEST2014_DATASET: Path = Path("/mnt/d/Dataset/mscoco/test2014")
    COCO_VAL2014_DATASET: Path = Path("/mnt/d/Dataset/mscoco/val2014")

    COCO_TRAIN2014_CAPTIONS_JSON: Path = Path(
        "/mnt/d/Dataset/mscoco/annotations/captions_train2014.json"
    )
    COCO_VAL2014_CAPTIONS_JSON: Path = Path(
        "/mnt/d/Dataset/mscoco/annotations/captions_val2014.json"
    )
    COCO_TRAIN2014_CAPTIONS_JA_JSON: Path = Path(
        "/mnt/d/Dataset/mscoco/annotations/stair_captions_v1.2_train.json"
    )
    COCO_VAL2014_CAPTIONS_JA_JSON: Path = Path(
        "/mnt/d/Dataset/mscoco/annotations/stair_captions_v1.2_val.json"
    )
