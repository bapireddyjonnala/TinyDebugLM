import torch.nn as nn

from models.attention import MultiHeadSelfAttention
from models.feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embedding_dim,
            num_heads
        )

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.ffn = FeedForward(
            embedding_dim,
            hidden_dim
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attention = self.attention(x)

        x = self.norm1(
            x + self.dropout(attention)
        )

        feedforward = self.ffn(x)

        x = self.norm2(
            x + self.dropout(feedforward)
        )

        return x