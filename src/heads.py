"""
heads.py — Multi-task auxiliary heads:
  1. CategoryClassifierHead  — spatial-aware transformer on quantized features → class logits
  2. GeometricFingerprintHead — attention-weighted code aggregation → contrastive embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ClassifierConfig, FingerprintConfig


class CategoryClassifierHead(nn.Module):
    """
    Codebook-aware semantic classifier: learns semantic meaning for each discrete code,
    uses spatial attention to understand code relationships, and aggregates with
    learned importance weights. Designed to leverage VQ discrete structure.
    """

    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.codebook_size = cfg.codebook_size
        self.num_classes = cfg.num_classes
        
        # Learn semantic meaning for each discrete code
        self.code_embeddings = nn.Embedding(cfg.codebook_size, cfg.hidden_dim)
        
        # Spatial position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim),
        )
        
        # Learn spatial relationships between codes
        self.spatial_attention = nn.MultiheadAttention(
            cfg.hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(cfg.hidden_dim)
        
        # Class-specific code importance
        self.code_importance = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, code_indices: torch.Tensor, mask_scores: torch.Tensor,
                voxel_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            code_indices:  [B, N]    long  — discrete code indices from VQ
            mask_scores:   [B, N]    float — geometric importance scores
            voxel_coords:  [B, N, 3] float — 3D voxel coordinates
        Returns:
            logits: [B, num_classes]
        """
        B, N = code_indices.shape
        
        # 1. Get semantic meaning for each code
        code_semantics = self.code_embeddings(code_indices)  # [B, N, D]
        
        # 2. Add spatial position information
        pos_encoding = self.pos_encoder(voxel_coords)  # [B, N, D]
        code_semantics = code_semantics + pos_encoding
        
        # 3. Weight by geometric importance from masker
        weighted_semantics = code_semantics * mask_scores.unsqueeze(-1)  # [B, N, D]
        
        # 4. Learn spatial relationships between codes via attention
        spatial_context, attn_weights = self.spatial_attention(
            weighted_semantics, weighted_semantics, weighted_semantics
        )  # [B, N, D]
        spatial_context = self.attn_norm(spatial_context + weighted_semantics)  # Residual
        
        # 5. Learn which codes are most important for classification
        code_weights = self.code_importance(spatial_context).squeeze(-1)  # [B, N]
        code_weights = torch.softmax(code_weights, dim=1)  # Normalize to sum=1
        
        # 6. Aggregate weighted semantic information
        global_repr = (spatial_context * code_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        
        # 7. Classify
        return self.classifier(global_repr)


class GeometricFingerprintHead(nn.Module):
    """
    Attention-weighted aggregation of quantized features → normalized fingerprint.
    Used with InfoNCE contrastive loss.
    """

    def __init__(self, cfg: FingerprintConfig):
        super().__init__()
        self.temperature = cfg.temperature
        self.code_attn = nn.Sequential(
            nn.Linear(cfg.codebook_dim, cfg.codebook_dim // 2),
            nn.Tanh(),
            nn.Linear(cfg.codebook_dim // 2, 1),
        )
        self.projector = nn.Sequential(
            nn.Linear(cfg.codebook_dim, cfg.fingerprint_dim),
            nn.LayerNorm(cfg.fingerprint_dim),
        )

    def forward(self, quant_kept: torch.Tensor):
        """
        Args:
            quant_kept: [B, K, d_q]  — quantized features at kept positions
        Returns:
            fingerprint: [B, fingerprint_dim]  normalized
            attn_weights: [B, K]
        """
        attn_logits = self.code_attn(quant_kept).squeeze(-1)         # [B, K]
        attn_weights = F.softmax(attn_logits / self.temperature, dim=1)  # [B, K]
        weighted = (quant_kept * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, d_q]
        fingerprint = F.normalize(self.projector(weighted), dim=-1)      # [B, fp_dim]
        return fingerprint, attn_weights


def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss between two views of the same batch.
    Args:
        z_i, z_j: [B, D]  — normalized fingerprint embeddings
    Returns:
        scalar loss
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    logits_ij = torch.mm(z_i, z_j.T) / temperature   # [B, B]
    logits_ji = logits_ij.T
    labels = torch.arange(z_i.size(0), device=z_i.device)
    loss = (F.cross_entropy(logits_ij, labels) + F.cross_entropy(logits_ji, labels)) / 2
    return loss
