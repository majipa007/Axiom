import logging
from typing import Dict
import sentencepiece as smp
import json
from config.config import Config
from src.training.dataset import download_and_store
from config.logging_config import setup_logging
from src.training.token import tokenizer_download

cnf = Config()
logger = logging.getLogger(__name__)


def main():
    # ======================= Data set Loading =======================
    try:
        logger.info("loading the dataset")
        if not cnf.dataset_file.exists():
            logger.debug("dataset not found, downloading and storing the data...")
            download_and_store()
            logger.debug("downloaded and stored the data")
        with open(cnf.dataset_file) as f:
            logger.debug("reading the data")
            dataset: Dict = json.load(f)
        logger.info("completed reading the data")
    except Exception as exe:
        logger.error(f"error loading dataset: {exe}")
        return

    # ======================= Tokenizer loading =======================
    try:
        logger.info("loading the tokenizer")
        if not cnf.tokenizer_file_dir.exists():
            logger.debug("tokenizer not found, downloading tokenizer")
            tokenizer_download()
        tokenizer = smp.SentencePieceProcessor(model_file=str(cnf.tokenizer_file_dir))
        tokens = tokenizer.encode(str(dataset["train"][0]))
    except Exception as exe:
        logger.error(f"error loading the tokentizer: {exe}")
        return


if __name__ == "__main__":
    setup_logging()
    main()
