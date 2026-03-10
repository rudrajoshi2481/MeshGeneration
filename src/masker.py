"""
masker.py — Geometric Importance Scorer (3D Masker).
Scores all voxel positions, selects top-k, returns kept/masked split.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import MaskerConfig


class GeometricMasker3D(nn.Module):
    """
    Inputs: grid_feat [B, C, R, R, R], grid_mask [B, R, R, R]
    Steps:
        1. Flatten grid → [B, R^3, C]
        2. Score each voxel position → importance score in [0,1]
        3. Project to codebook_dim
        4. Return top-k kept and remaining indices + scores
    """

    def __init__(self, cfg: MaskerConfig):
        super().__init__()
        self.topk_ratio = cfg.topk_ratio

        # Geometric importance scorer
        self.scorer = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.input_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.input_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Pre-project to codebook dim
        self.pre_proj = nn.Linear(cfg.input_dim, cfg.codebook_dim, bias=False)

        # Class-conditional bias (optional, adds learned per-class offset)
        self.class_bias = nn.Embedding(cfg.num_classes, 1)
        nn.init.zeros_(self.class_bias.weight)

        self.norm = nn.LayerNorm(cfg.input_dim, elementwise_affine=False)

    def forward(self, grid_feat: torch.Tensor, grid_mask: torch.Tensor,
                labels: torch.Tensor = None) -> Dict:
        """
        Args:
            grid_feat: [B, C, R, R, R]
            grid_mask: [B, R, R, R]   bool
            labels:    [B]            optional class labels for conditional bias
        Returns dict:
            all_features:   [B, R^3, codebook_dim]  — projected features for ALL positions
            score_map:      [B, R^3]                 — importance scores
            sample_index:   [B, K]                   — indices of KEPT voxels
            remain_index:   [B, R^3-K]               — indices of MASKED voxels
            grid_mask_flat: [B, R^3]                 — original occupancy mask
        """
        B, C, R, _, _ = grid_feat.shape
        N = R * R * R

        # [B, C, R, R, R] → [B, R^3, C]
        feat_flat = grid_feat.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        mask_flat = grid_mask.reshape(B, N)  # [B, N] bool

        # Normalize features
        feat_norm = self.norm(feat_flat)

        # Score every position
        scores = self.scorer(feat_norm).squeeze(-1)  # [B, N]

        # Add class-conditional bias if labels provided
        if labels is not None:
            bias = self.class_bias(labels)  # [B, 1]
            scores = (scores + bias).sigmoid()

        # Zero out empty voxels so they are never selected
        scores = scores * mask_flat.float()

        # Project to codebook dim
        all_features = self.pre_proj(feat_norm)  # [B, N, codebook_dim]

        # Top-k selection
        K = max(1, int(N * self.topk_ratio))
        _, sorted_idx = scores.sort(dim=1, descending=True)
        sample_index = sorted_idx[:, :K]    # [B, K]  — kept
        remain_index = sorted_idx[:, K:]    # [B, N-K] — masked

        return {
            "all_features": all_features,
            "score_map": scores,
            "sample_index": sample_index,
            "remain_index": remain_index,
            "grid_mask_flat": mask_flat,
        }
