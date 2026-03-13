"""
attention.py — Multi-head self-attention block used by the De-Masker transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with pre-LayerNorm."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    [B, T, D]
            mask: [B, T] bool — True means IGNORE (padding)
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)           # each [B, T, H, Hd]
        q = q.transpose(1, 2)                  # [B, H, T, Hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        if mask is not None:
            # mask: [B, T] → [B, 1, 1, T]
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        out = self.proj(out)
        return residual + out


class FFN(nn.Module):
    """Feed-forward network with pre-LayerNorm and GELU."""

    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TransformerBlock(nn.Module):
    """MHSA + FFN block."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ffn = FFN(dim, ff_mult=4, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.attn(x, mask)
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks."""

    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
