import pathlib
from datetime import datetime

import torch
from torch._prims_common import check
from config.config import Config
from src.training.batching import batch_data
from src.model.model import  Axiom
from torch import nn
from pathlib import Path

class Trainer():
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        conf = Config()
        self.batch_size = conf.batch_size
        self.context = conf.context_length
        self.vocab_size = conf.vocab_size
        self.train_dir = conf.token_train_dir
        self.val_dir = conf.token_val_dir
        self.model = Axiom(
            vocab_size=conf.vocab_size,
            context_length=conf.context_length,
            d_model=conf.d_model,
            n_heads=conf.n_heads,
            n_blocks=conf.n_blocks,
                    ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = 3e-4,
        )

    def estimate_loss(self, path, steps):
        loss_sum = 0
        self.model.eval()
        with torch.no_grad():
            for step in range(steps):
                batch = batch_data(path, self.context, self.batch_size)
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                y = y.reshape(-1)
                output = self.model(x)
                output = output.reshape(-1, self.vocab_size)
                loss = self.loss_fn(output, y)
                loss_sum += loss.item()
        self.model.train()
        return loss_sum/(steps)



    def train(self, steps: int, start_step: int = 0):
        for step in range(start_step, steps):
            batch = batch_data(self.train_dir,self.context, self.batch_size)
            self.optimizer.zero_grad()
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            output = self.model(x)
            output = output.reshape(-1, self.vocab_size)
            y = y.reshape(-1)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            if (step + 1) % 500 == 0:
                self.save_checkpoint(
                    step=step + 1,
                    train_loss=loss.item(),
                    val_loss=self.estimate_loss(self.val_dir, 10),
                )
            if step%10==0:
                train_loss = self.estimate_loss(self.train_dir, 20)
                val_loss = self.estimate_loss(self.val_dir, 20)
                print(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"Step {step} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )


    def save_checkpoint(self, step: int, train_loss: float, val_loss:float):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        torch.save(checkpoint, checkpoint_dir / "axiom_latest.pt")

    def load_checkpoint(self, path="checkpoints/axiom_latest.pt"):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["step"]
