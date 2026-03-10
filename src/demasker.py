"""
demasker.py — Volumetric De-Masker.
Takes quantized KEPT tokens + mask token → transformer → full grid features.
"""

import torch
import torch.nn as nn

from attention import TransformerEncoder
from config import DemaskerConfig


class VolumetricDemasker3D(nn.Module):
    """
    Given quantized features at KEPT positions + mask tokens at REMAIN positions,
    run a transformer to recover full-resolution features.

    Inputs:
        quant_kept:    [B, K, d_q]     — quantized features at kept voxels
        kept_idx:      [B, K]          — flat spatial indices of kept voxels
        remain_idx:    [B, M]          — flat spatial indices of masked voxels
        grid_mask:     [B, N]          — bool, True = occupied (for padding mask)
    Output:
        full_feat:     [B, N, output_dim]  where N = R^3
    """

    def __init__(self, cfg: DemaskerConfig, grid_size: int):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        N = grid_size ** 3

        # Learnable mask token (fills unselected positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.input_dim) * 0.02)

        # Positional embedding for every voxel in the grid
        self.pos_embed = nn.Embedding(N, cfg.input_dim)

        # Transformer
        self.transformer = TransformerEncoder(
            dim=cfg.input_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

        # Project back to decoder-expected dimension
        self.out_proj = nn.Linear(cfg.input_dim, cfg.output_dim)

    def forward(
        self,
        quant_kept: torch.Tensor,
        kept_idx: torch.Tensor,
        remain_idx: torch.Tensor,
        grid_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, K, d_q = quant_kept.shape
        M = remain_idx.shape[1]
        N = K + M
        device = quant_kept.device

        # Start with mask tokens for all N positions
        tokens = self.mask_token.expand(B, N, -1).clone()  # [B, N, d_q]

        # Place quantized kept features at their positions
        # We need to scatter kept into the right slots
        # Reconstruct full ordering: sort by original index
        all_idx = torch.cat([kept_idx, remain_idx], dim=1)   # [B, N]
        # inverse permutation: where does each position end up?
        inv_perm = torch.argsort(all_idx, dim=1)              # [B, N]

        # Fill kept positions
        tokens_kept = quant_kept                               # [B, K, d_q]
        tokens_masked = self.mask_token.expand(B, M, -1)      # [B, M, d_q]
        tokens_cat = torch.cat([tokens_kept, tokens_masked], dim=1)  # [B, N, d_q]

        # Reorder to spatial order
        tokens = tokens_cat.gather(
            1, inv_perm.unsqueeze(-1).expand(-1, -1, d_q)
        )  # [B, N, d_q]

        # Add positional embeddings using the spatial indices
        pos_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, N]
        # Reconstruct original spatial indices from all_idx sorted
        spatial_idx = all_idx.gather(1, inv_perm)             # = sorted all_idx = 0..N-1
        tokens = tokens + self.pos_embed(spatial_idx)

        # Padding mask: True = ignore (non-occupied voxels)
        pad_mask = ~grid_mask  # [B, N]

        # Run transformer
        tokens = self.transformer(tokens, mask=pad_mask)       # [B, N, d_q]

        # Project to output dim
        return self.out_proj(tokens)                           # [B, N, output_dim]
