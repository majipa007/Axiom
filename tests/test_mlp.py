import torch
from src.model.feed_forward import MLP

x = torch.randn(2, 4, 256)

mlp = MLP(d_model=256)

output = mlp(x)

print("Input: ", x.shape)
print("Output:", output.shape)

loss = output.sum()
loss.backward()

for name, param in mlp.named_parameters():
    print(
        name,
        "✅ gradient OK" if param.grad is not None else "❌ NO GRADIENT"
    )
