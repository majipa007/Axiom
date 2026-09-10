import json
from pathlib import Path
from datasets import load_dataset

from config.config import Config

cnf = Config()


def download_and_store():
    ds = load_dataset(cnf.dataset)
    data = {split: dataset.to_list() for split, dataset in ds.items()}
    path = Path(cnf.dataset_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
