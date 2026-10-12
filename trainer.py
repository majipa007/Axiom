from torch._higher_order_ops import out_dtype
from src.training.tokenize import detokenize

from src.training.train import Trainer
from src.inference.generate import Generator
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

training = Trainer()
training.train(300)


generator = Generator(
    model=training.model,
    context_length=training.context,
)

x = torch.tensor([[1, 25, 42]], device=training.device)

output = generator.generate(x, max_new_tokens=20)
token_ids = output[0].cpu().tolist()
print(detokenize(token_ids))
