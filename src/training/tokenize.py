import logging
from config.config import Config
from huggingface_hub import hf_hub_download
import sentencepiece as spm
from pathlib import Path
from config.logging_config import setup_logging
import numpy as np

from src.errors import tokenization_error

cnf = Config()

logger = logging.getLogger(__name__)


def tokenizer_download():
    hf_hub_download(
        repo_id=cnf.tokenizer_repo_id,
        filename=cnf.tokenizer_file_name,
        local_dir=cnf.tokenizer_local_dir,
    )


def tokenize_data(dataset: dict) -> None:
    try:
        if not cnf.token_train_dir.exists():
            logger.info("no tokens found for training so starting to tokenize")
            if not cnf.tokenizer_file_dir.exists():
                tokenizer_download()
            tokenizer = spm.SentencePieceProcessor(
                model_file=str(cnf.tokenizer_file_dir)
            )
            cnf.token_dir.mkdir(parents=True, exist_ok=True)

            tokenize_split(
                dataset=dataset["train"],
                tokenizer=tokenizer,
                output_file=str(cnf.token_train_dir),
            )

            tokenize_split(
                dataset=dataset["validation"],
                tokenizer=tokenizer,
                output_file=str(cnf.token_val_dir),
            )
        else:
            logger.info("using cached tokens for training")
    except Exception as exe:
        raise tokenization_error("Error creating tokenize data") from exe


def tokenize_split(
    dataset: list,
    tokenizer: spm.SentencePieceProcessor,
    output_file: str,
) -> None:
    logger.debug("Saving tokens in the file.")
    all_tokens = []
    with open(output_file, "wb") as f:
        for sample in dataset:
            text = sample["text"]

            tokens = tokenizer.encode(text)

            tokens.append(tokenizer.eos_id())

            np.asarray(tokens, dtype=np.uint16).tofile(f)
    logger.debug(
        "saved tokens to %s",
        output_file,
    )
