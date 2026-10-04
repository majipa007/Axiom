import torch
from config.config import Config
from src.training.batching import batch_data
from src.model.model import  Axiom
from torch import nn

class Trainer():
    def __init__(self) -> None:
        conf = Config()
        self.batch_size = conf.batch_size
        self.context = conf.context_length
        self.vocab_size = conf.vocab_size
        self.model = Axiom(
            vocab_size=conf.vocab_size,
            context_length=conf.context_length,
            d_model=conf.d_model,
            n_heads=conf.n_heads,
            n_blocks=conf.n_blocks,
                    )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = 3e-4,
        )


    def train(self, steps: int):
        for step in range(steps):
            batch = batch_data(self.context, self.batch_size)
            self.optimizer.zero_grad()
            x = batch["x"]
            y = batch["y"]

            output = self.model(x)
            output = output.reshape(-1, self.vocab_size)
            y = y.reshape(-1)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            if step%10==0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

