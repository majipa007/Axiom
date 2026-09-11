import logging
from config.config import Config
from src.errors import batching_error, data_loading_error, tokenization_error
from src.training.dataset import load_data
from src.training.tokenize import tokenize_data
from src.training.batching import batch_data
from config.logging_config import setup_logging

cnf = Config()
logger = logging.getLogger(__name__)


def main():
    # ======================= Data set Loading =======================
    try:
        logger.info("dataset loading")
        dataset: dict = load_data()
        logger.info("dataset loading completed")

        # ======================= Tokenizer loading =======================
        logger.info("tokenization")
        tokenize_data(dataset)
        logger.info("tokenization completed")

        # ======================= Batch loading =======================
        logger.info("batching")
        batches = batch_data(cnf.context_length, cnf.batch_size)
        logger.info("batching completed")

    except data_loading_error as exe:
        logger.error(f"error while loading the dataset: {exe}")
        raise

    except tokenization_error as exe:
        logger.error(f"error while tokenization: {exe}")
        raise

    except batching_error as exe:
        logger.error(f"error while batching the dataset: {exe}")
        raise

    except Exception as exe:
        logger.error(f"Unknown error in the pipeline: {exe}")
        raise


if __name__ == "__main__":
    setup_logging()
    main()
