import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.
    """

    def __init__(self, max_length: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        positions = positions.expand(batch_size, seq_len)

        return self.embedding(positions)