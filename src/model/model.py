import torch
import torch.nn as nn


class Axiom(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # V no of items in the vocab
        context_length: int,  # T in BTC this is the time dimension
        d_model: int,  # C, how many dimensions each token will have
    ) -> None:
        super().__init__()

        self.context_length = context_length

        # [V, C]
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # [T, C]
        self.position_embedding = nn.Embedding(context_length, d_model)
