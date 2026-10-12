import torch

from src.training.train import Trainer
from src.training.tokenize import detokenize
from src.inference.generate import Generator


def main():
    trainer = Trainer()

    start_step = trainer.load_checkpoint()
    trainer.train(steps=50000, start_step=start_step)

    generator = Generator(
        model=trainer.model,
        context_length=trainer.context,
    )

    x = torch.tensor([[1, 25, 42]], device=trainer.device)

    output = generator.generate(x, max_new_tokens=20)
    token_ids = output[0].cpu().tolist()

    print("Axiom:", detokenize(token_ids))


if __name__ == "__main__":
    main()
