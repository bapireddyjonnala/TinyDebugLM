import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention with Causal Mask
    """

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear projections
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.out = nn.Linear(embedding_dim, embedding_dim)

        # Attention dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, seq_len, embed_dim = x.shape

        # Compute Query, Key, Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Shape:
        # (batch, heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled Dot Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # -----------------------------
        # Causal Mask
        # -----------------------------
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Attention weights
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Attention output
        attention = torch.matmul(weights, V)

        # Merge heads
        attention = attention.transpose(1, 2).contiguous()

        attention = attention.view(
            batch_size,
            seq_len,
            embed_dim
        )

        # Final projection
        output = self.out(attention)

        return output