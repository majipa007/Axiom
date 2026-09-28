import torch
from src.model.block import TransformerBlock

block = TransformerBlock(
    d_model=256,
    n_heads=4,
    context_length=128,
)

x = torch.randn(2, 4, 256)

output = block(x)

print("Input: ", x.shape)
print("Output:", output.shape)

loss = output.sum()
loss.backward()

for name, param in block.named_parameters():
    if param.grad is None:
        print(name, "❌ NO GRADIENT")
    else:
        print(name, "✅ gradient OK")
