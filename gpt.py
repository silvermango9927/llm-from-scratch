from config import gpt_config
from transformer import TransformerBlock

import torch

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
transformer_block = TransformerBlock(gpt_config)
output = transformer_block(x)
print(output.shape, x.shape)