from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    local_dir: Path = project_root / ".local"
    dataset_dir: Path = local_dir / "dataset"
    dataset_file: Path = dataset_dir / "dataset.txt"
    

    # Data
    dataset: str = "tiny_shakespeare"


