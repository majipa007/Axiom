from config.config import Config
from huggingface_hub import hf_hub_download


cnf = Config()


def tokenizer_download():
    hf_hub_download(
        repo_id=cnf.tokenizer_repo_id,
        filename=cnf.tokenizer_file_name,
        local_dir=cnf.tokenizer_local_dir,
    )
