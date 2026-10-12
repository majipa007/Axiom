import torch
from torch import nn 

class Generator:
    def __init__(self, model: nn.Module, context_length: int) -> None:
        self.model = model
        self.context_length = context_length

    def generate(self, x:torch.Tensor, max_new_tokens: int):
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_context = x[:, -self.context_length: ]
                logits =  self.model(x_context)
                last_logits = logits[:, -1, :]
                probabilities = torch.softmax(last_logits, dim = -1)
                next_token = torch.argmax(probabilities, dim = -1, keepdim=True)
                x = torch.cat((x, next_token), dim=1)
        return x



