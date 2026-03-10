"""
quantizer.py — EMA Vector Quantization with random restart of dead codes.
Straight-Through Estimator gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import VQConfig


class VQEmbeddingEMA(nn.Module):
    """
    Vector Quantization with EMA codebook updates and random restart.

    Forward returns:
        quantized   [B*N, D]  — quantized vectors (straight-through gradient)
        vq_loss     scalar    — commitment loss only (codebook updated via EMA)
        indices     [B*N]     — codebook indices
    """

    def __init__(self, cfg: VQConfig):
        super().__init__()
        K = cfg.num_embeddings
        D = cfg.embedding_dim
        self.num_embeddings = K
        self.embedding_dim = D
        self.commitment_weight = cfg.commitment_weight
        self.decay = cfg.ema_decay
        self.threshold_ema_dead_code = cfg.threshold_ema_dead_code

        # Codebook — not a parameter (updated via EMA)
        embed = torch.randn(K, D)
        embed = F.normalize(embed, dim=1)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(K))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("_initted", torch.zeros(1))

    @torch.no_grad()
    def _ema_update(self, flat_input: torch.Tensor, encoding_indices: torch.Tensor):
        """Update codebook via exponential moving average."""
        K = self.num_embeddings
        D = self.embedding_dim
        device = flat_input.device

        # One-hot encode indices
        one_hot = torch.zeros(flat_input.shape[0], K, device=device)
        one_hot.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Cluster counts
        new_cluster_size = one_hot.sum(dim=0)  # [K]
        self.cluster_size.mul_(self.decay).add_(new_cluster_size * (1 - self.decay))

        # Cluster embedding sums
        dw = one_hot.t() @ flat_input  # [K, D]
        self.embed_avg.mul_(self.decay).add_(dw * (1 - self.decay))

        # Normalize
        n = self.cluster_size.unsqueeze(1)
        self.embed = self.embed_avg / (n + 1e-8)

    @torch.no_grad()
    def _random_restart(self, flat_input: torch.Tensor):
        """Replace dead codes with random encoder outputs."""
        dead = self.cluster_size < self.threshold_ema_dead_code
        n_dead = dead.sum().item()
        if n_dead == 0:
            return
        # Sample random encoder outputs
        n = flat_input.shape[0]
        random_idx = torch.randint(0, n, (int(n_dead),), device=flat_input.device)
        self.embed[dead] = flat_input[random_idx].detach()
        self.embed_avg[dead] = flat_input[random_idx].detach()
        self.cluster_size[dead] = 1.0

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [N, D]  — any number of vectors
        Returns:
            quantized: [N, D]
            loss:      scalar
            indices:   [N] long
        """
        input_dtype = inputs.dtype
        # Codebook buffers are float32; cast inputs for distance computation
        flat = inputs.reshape(-1, self.embedding_dim).float()

        # Distances: ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )  # [N, K]

        encoding_indices = dist.argmin(dim=1)  # [N]
        quantized = self.embed[encoding_indices]  # [N, D]

        # EMA update (training only)
        if self.training:
            self._ema_update(flat.detach(), encoding_indices)
            self._random_restart(flat.detach())

        # Commitment loss
        loss = self.commitment_weight * F.mse_loss(quantized.detach(), flat)

        # Straight-through estimator
        quantized_st = flat + (quantized - flat).detach()
        quantized_st = quantized_st.reshape(inputs.shape).to(input_dtype)

        return quantized_st, loss, encoding_indices

    def codebook_utilization(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in this batch."""
        return indices.unique().numel() / self.num_embeddings

    def perplexity(self, indices: torch.Tensor) -> float:
        """Effective number of codebook entries used (exponential of entropy)."""
        counts = torch.bincount(indices.flatten(), minlength=self.num_embeddings).float()
        probs = counts / counts.sum()
        entropy = -(probs * (probs + 1e-10).log()).sum()
        return entropy.exp().item()
