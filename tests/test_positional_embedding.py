import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from models.positional_embedding import PositionalEmbedding

pos_embedding = PositionalEmbedding(
    max_length=512,
    embedding_dim=256
)

x = torch.randint(0, 100, (2, 10))

y = pos_embedding(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)