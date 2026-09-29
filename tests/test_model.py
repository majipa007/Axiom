from src.model.model import Axiom
import torch
vocab_size = 8000

model = Axiom(
    vocab_size=vocab_size,
    context_length=128,
    d_model=256,
    n_heads=4,
    n_blocks=4,
)
x = torch.randint(
    0,
    vocab_size,
    (2, 4),
)
output = model(x)

print("Input:", x.shape)
print("Output:", output.shape)


loss = output.sum()
loss.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"{name} ❌ NO GRADIENT")
    else:
        print(f"{name} ✅ gradient OK")
