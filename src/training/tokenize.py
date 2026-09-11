import logging
from config.config import Config
from huggingface_hub import hf_hub_download
import sentencepiece as spm
from pathlib import Path
from config.logging_config import setup_logging
import numpy as np

cnf = Config()

logger = logging.getLogger(__name__)


def tokenizer_download():
    hf_hub_download(
        repo_id=cnf.tokenizer_repo_id,
        filename=cnf.tokenizer_file_name,
        local_dir=cnf.tokenizer_local_dir,
    )


def tokenize_data(dataset: dict) -> None:
    if not cnf.token_dir.exists():
        logger.info("no tokens found for training so starting to tokenize")
        if not cnf.tokenizer_file_dir.exists():
            tokenizer_download()
        tokenizer = spm.SentencePieceProcessor(model_file=str(cnf.tokenizer_file_dir))
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


def tokenize_split(
    dataset: list,
    tokenizer: spm.SentencePieceProcessor,
    output_file: str,
) -> None:
    logger.debug("Saving tokens in the file.")
    all_tokens = []
    for sample in dataset:
        text = sample["text"]

        tokens = tokenizer.encode(text)

        all_tokens.extend(tokens)

        all_tokens.append(tokenizer.eos_id())

    tokens_array = np.array(all_tokens, dtype=np.uint16)
    tokens_array.tofile(output_file)
    logger.debug(
        "saved %d tokens to %s",
        len(tokens_array),
        output_file,
    )
