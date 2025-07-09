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
    
def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)\
    
    return idx

torch.manual_seed(123)
model = GPTModel(gpt_config)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

start_context = "Hello, I am"
encoded = tokenizer_bpe.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval() #A
out = generate_text(
model=model,
idx=encoded_tensor,
max_new_tokens=6,
context_size=gpt_config["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer_bpe.decode(out.squeeze(0).tolist())
print(decoded_text)