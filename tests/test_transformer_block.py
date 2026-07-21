import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.transformer_block import TransformerBlock

block = TransformerBlock(
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024
)

x = torch.randn(2, 10, 256)

y = block(x)

print("Input :", x.shape)
print("Output:", y.shape)