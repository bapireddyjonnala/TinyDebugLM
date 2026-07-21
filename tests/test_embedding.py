import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from models.embeddings import TokenEmbedding
embedding = TokenEmbedding(
    vocab_size=16000,
    embedding_dim=256
)

x = torch.tensor([[1, 25, 100, 999]])

y = embedding(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)