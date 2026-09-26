import torch

from src.model.attention import SingleHeadAttention


torch.manual_seed(42)

B = 2
T = 4
C = 8
H = 2

attention = SingleHeadAttention(
    d_model=C,
    head_size=H,
    context_length=8,
)

attention.eval()

x = torch.randn(B, T, C)
output = attention(x)
print(output)

# 1. Shape check
assert output.shape == (B, T, H)
print("Output shape:", output.shape)

# 2. Check for NaN or infinity
assert torch.isfinite(output).all()
print("All output values are finite")

# 3. Causal-mask check
# Change only the final/future token
changed_x = x.clone()
changed_x[:, -1, :] += 100

changed_output = attention(changed_x)

# Earlier tokens must remain unchanged
assert torch.allclose(
    output[:, :-1, :],
    changed_output[:, :-1, :],
    atol=1e-5,
)

print("Future token cannot affect earlier tokens")

# 4. Gradient-flow check
output.sum().backward()

for name, parameter in attention.named_parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    print(f"{name}: gradient OK")

print("Single-head attention passed all checks")
