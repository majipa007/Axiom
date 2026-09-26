import torch 
from torch import nn 

class SingleHeadAttention(nn.Module):
    def __init__(self, 
                 d_model:int, # dimention of the model's training vectors
                 head_size: int, # head size of the single head attnetion 
                 ) -> None:
        super().__init__()
        self.head_size = head_size
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
        query = self.query(x) # query = x.WQT
        # x : [B T C] | WQT: [C H]
        # query = [B T H]
        key = self.key(x)
        value = self.value(x)

        # [B T H] @ [B H T] ->  [B T T]
        # scores = Query@Key(Transpose)
        scores = query @ key.Transpose(-2, -1)
        scaled_scores = scores/(self.head_size ** 0.5 )


        

        return scores 

