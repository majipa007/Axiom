import torch 
import torch.nn as nn 

class SingleHeadAttention(nn.Module):
    def __init__(self, 
                 d_model:int, # dimention of the model's training vectors
                 head_size: int, # head size of the single head attnetion 
                 ) -> None:
        super().__init__()
        self.query = nn.Linear(
            d_model, 
            head_size, 
            bias=False
        )

        self.key = nn.Linear(
            d_model,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            d_model,
            head_size,
            bias=False
        )

    

    def forward(self, x:torch.Tensor):
        # x = [B, T, C]
        # [B, T, C] gets converted to [B, T, head_size]

