from random import randint
import numpy as np
from config.config import Config
from src.errors import batching_error
import torch
from pathlib import Path

cnf = Config()


def load_tokens(path):
    data = np.memmap(path, dtype=np.uint16, mode="r")

    return data


def batch_data(data_path:Path, context_length: int, batch_size: int) -> dict:
    try:
        x = []
        y = []
        data = load_tokens(data_path)
        for i in range(batch_size):
            start = randint(0, len(data) - context_length - 1)
            x.append(data[start : start + context_length])
            y.append(data[start + 1 : start + context_length + 1])
        X = torch.from_numpy(np.stack(x)).long()
        Y = torch.from_numpy(np.stack(y)).long()
        return {"x": X, "y": Y}
    except Exception as exe:
        raise batching_error("Failed to create batch") from exe

