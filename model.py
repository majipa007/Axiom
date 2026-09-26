import torch.nn as nn 

class Axiom(nn.Module):

    def __init__(self, vocab_size, context_length, d_model) -> None:
        super().__init__()
        
        self.tokenize_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(context_length, d_model)
