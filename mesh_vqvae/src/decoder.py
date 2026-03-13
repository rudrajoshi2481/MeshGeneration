"""
decoder.py — Occupancy decoder.
Given full grid features [B, N, C] and query points [B, M, 3],
predict occupancy via trilinear interpolation + MLP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DecoderConfig


class OccupancyDecoder(nn.Module):
    """
    1. Reshape full features back to [B, C, R, R, R]
    2. Trilinear interpolation at query point locations
    3. MLP to predict occupancy logits
    """

    def __init__(self, cfg: DecoderConfig, grid_res: int):
        super().__init__()
        self.grid_res = grid_res
        C = cfg.grid_channels

        dims = [C] + cfg.hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(cfg.dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, full_feat: torch.Tensor, query_pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            full_feat:  [B, N, C]  where N = R^3
            query_pts:  [B, M, 3]  query points in [-0.5, 0.5]
        Returns:
            logits: [B, M]
        """
        B, M, _ = query_pts.shape
        R = self.grid_res
        C = full_feat.shape[-1]

        # Reshape to 3D grid
        grid = full_feat.reshape(B, R, R, R, C).permute(0, 4, 1, 2, 3)  # [B, C, R, R, R]

        # Normalize query pts to [-1, 1] for grid_sample
        # Points are in [-0.5, 0.5] → divide by 0.5
        pts_norm = query_pts / 0.5  # [-1, 1]
        pts_norm = pts_norm.clamp(-1, 1)

        # grid_sample expects [B, D, H, W, 3] (5D) → add extra dim
        pts_for_sample = pts_norm.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, M, 3]

        # Trilinear interpolation
        interp = F.grid_sample(
            grid,
            pts_for_sample,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, C, 1, 1, M]
        interp = interp.squeeze(2).squeeze(2).permute(0, 2, 1)  # [B, M, C]

        # MLP → logits
        logits = self.mlp(interp).squeeze(-1)  # [B, M]
        return logits
