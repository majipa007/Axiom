import json
from pathlib import Path
from datasets import load_dataset
import logging
from config.config import Config
from errors import data_loading_error

cnf = Config()

logger = logging.getLogger()


def download_and_store() -> None:
    ds = load_dataset(cnf.dataset)
    data = {split: dataset.to_list() for split, dataset in ds.items()}
    path = Path(cnf.dataset_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_data() -> dict:
    try:
        if not cnf.dataset_file.exists():
            logger.debug("dataset not found, downloading and storing the data...")
            download_and_store()
            logger.debug("downloaded and stored the data")
        with open(cnf.dataset_file) as f:
            logger.debug("reading the data")
            dataset: dict = json.load(f)
            return dataset
    except Exception as exe:
        raise data_loading_error("Failed to load data") from exe
