import logging
from typing import Dict

from fsspec.spec import json
from config.config import Config
from src.training.dataset import download_and_store
from config.logging_config import setup_logging

cnf = Config()
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("starting the function")
        if not cnf.dataset_file.exists():
            logger.info("dataset not found, downloading and storing the data...")
            download_and_store()
            logger.info("downloaded and stored the data")
        with open(cnf.dataset_file) as f:
            logger.info("reading the data")
            dataset: Dict = json.load(f)
            logger.info("completed reading the data")
        print(dataset["train"][0])
    except Exception as exe:
        logger.error(f"error: {exe}")


if __name__ == "__main__":
    setup_logging()
    main()
