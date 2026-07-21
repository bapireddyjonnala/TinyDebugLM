import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from models.feedforward import FeedForward

ffn = FeedForward(
    embedding_dim=256,
    hidden_dim=1024
)

x = torch.randn(2, 10, 256)

y = ffn(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)