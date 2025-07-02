from config import gpt_config
from transformer import TransformerBlock, LayerNormalization

import torch, torch.nn as nn

from concepts.tokenizer import tokenizer_bpe

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer_bpe.encode(txt1)))
batch.append(torch.tensor(tokenizer_bpe.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
transformer_block = TransformerBlock(gpt_config)
output = transformer_block(x)
print(output.shape, x.shape)

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNormalization(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.drop_emb(x)

        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
torch.manual_seed(123)
gpt_model = GPTModel(gpt_config)
output = gpt_model(batch)
print(output)  # Should be [batch_size, seq_length, vocab_size]
