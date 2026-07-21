import torch
import torch.nn as nn

from models.embeddings import TokenEmbedding
from models.positional_embedding import PositionalEmbedding
from models.transformer_block import TransformerBlock


class TinyDebugLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_length,
        embedding_dim=256,
        num_heads=8,
        hidden_dim=1024,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            max_length,
            embedding_dim
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads,
                hidden_dim,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        token = self.token_embedding(x)
        position = self.position_embedding(x)

        x = token + position

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits