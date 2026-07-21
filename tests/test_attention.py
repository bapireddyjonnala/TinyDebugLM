import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from models.attention import MultiHeadSelfAttention

attention = MultiHeadSelfAttention(
    embedding_dim=256,
    num_heads=8
)

x = torch.randn(2, 10, 256)

y = attention(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)