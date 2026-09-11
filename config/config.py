from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    local_dir: Path = project_root / ".local"
    dataset_dir: Path = local_dir / "dataset"
    dataset_file: Path = dataset_dir / "dataset.json"

    # Data
    dataset: str = "roneneldan/TinyStories"
    tokenizer_repo_id: str = "lakhera2023/gemma4-nano-tinystories"
    tokenizer_file_name: str = "tinystories_tokenizer.model"
    tokenizer_local_dir: Path = local_dir / "tokenizer"
    tokenizer_file_dir: Path = tokenizer_local_dir / tokenizer_file_name
    token_dir: Path = local_dir / "tokens/data"
    token_train_dir: Path = token_dir / "train.bin"
    token_val_dir: Path = token_dir / "validation.bin"

    # training params
    context_length: int = 100
    batch_size: int = 100
