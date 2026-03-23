"""
conditional_SEDD.py
--------------------
Modified copy of Yagna's SEDD.py for class-as-first-token conditioning.

Key changes from original SEDD.py:
  1. Vocabulary extended: 0-255 = mesh codes, 256 = MASK, 257-296 = class tokens
  2. Sequence format: [class_token, code_0, code_1, ..., code_4095] (length 4097)
  3. Class token is NEVER masked during noising — it's always kept fixed
  4. Generation: start with [class_token], model fills in the 4096 mesh tokens
  5. Removed separate class_embedding parameter (class is now part of the sequence)

Original: /data/joshi/yagnas_stuff/MLopsThesis/Models/SEDD.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Dict

NUM_MESH_CODES = 256       # 0-255 codebook indices
MASK_TOKEN_ID  = 256       # mask token for diffusion
NUM_CLASSES    = 40        # ModelNet40
CLASS_TOKEN_OFFSET = 257   # class c → token id (257 + c) # so the codebook indexes are from 0-256 so it dosen't interfeare with that
FULL_VOCAB_SIZE = 257 + NUM_CLASSES  # 297 total tokens


def class_to_token(c: torch.Tensor) -> torch.Tensor:
    """Convert class index (0-39) to token id (257-296)."""
    return c + CLASS_TOKEN_OFFSET


def token_to_class(t: torch.Tensor) -> torch.Tensor:
    """Convert token id (257-296) back to class index (0-39)."""
    return t - CLASS_TOKEN_OFFSET


class DiscreteNoiseSchedule(nn.Module):
    """Discrete diffusion noise schedule — masks tokens with MASK_TOKEN_ID."""

    def __init__(self, num_timesteps: int, mask_id: int, schedule_type: str = 'cosine'):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mask_id = mask_id

        if schedule_type == 'linear':
            self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif schedule_type == 'cosine':
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: mask tokens randomly according to noise level.
        Class token (position 0) is NEVER masked.

        Args:
            x_start: [B, L]  — L = 4097 (class_token + 4096 mesh codes)
            t:        [B]     — timestep per sample
        Returns:
            x_noisy:  [B, L]
        """
        noise_level = self.alphas_cumprod.to(x_start.device)[t]  # [B]
        noise = torch.rand(x_start.shape, device=x_start.device)

        # mask positions where noise > noise_level (higher t → more masking)
        mask = (noise >= noise_level.unsqueeze(-1))

        # NEVER mask position 0 (class token)
        mask[:, 0] = False

        x_noisy = x_start.clone()
        x_noisy[mask] = self.mask_id
        return x_noisy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = t.float() / 1000.0
        return self.time_embed(t_norm.unsqueeze(-1))  # [B, D]


class ConditionalSEDD(pl.LightningModule):
    """
    Discrete Diffusion Transformer with class-as-first-token conditioning.

    Sequence format during training and inference:
        [class_token | code_0 | code_1 | ... | code_4095]
         position 0    position 1 ... position 4096

    Vocabulary:
        0   - 255  : mesh codebook indices
        256        : MASK token
        257 - 296  : class tokens (257 = airplane, ..., 296 = door)

    Generation:
        1. Start with [class_token, MASK, MASK, ..., MASK]
        2. Iteratively unmask the 4096 mesh positions
        3. Class token at position 0 is never modified
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        schedule_type: str = 'cosine',
        learning_rate: float = 5e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = FULL_VOCAB_SIZE    # 297
        self.mask_id    = MASK_TOKEN_ID      # 256
        self.seq_len    = 4097               # 1 class token + 4096 mesh codes
        self.learning_rate = learning_rate

        # Noise schedule
        self.noise_schedule = DiscreteNoiseSchedule(
            num_timesteps=num_timesteps,
            mask_id=self.mask_id,
            schedule_type=schedule_type,
        )

        # Token + positional embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoder     = PositionalEncoding(d_model, self.seq_len + 10)

        # Time embedding
        self.time_embedding  = TimeEmbedding(d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection — only predicts mesh codes (0-255), not class tokens
        self.output_projection = nn.Linear(d_model, NUM_MESH_CODES)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4097]  token sequence (class token prepended)
            t: [B]        diffusion timesteps
        Returns:
            logits: [B, 4097, 256]  — predictions for each position (mesh vocab only)
        """
        token_embed = self.token_embedding(x)           # [B, L, D]
        token_embed = self.pos_encoder(token_embed)     # [B, L, D]

        time_embed  = self.time_embedding(t)            # [B, D]
        time_embed  = time_embed.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, L, D]

        hidden = token_embed + time_embed               # [B, L, D]
        hidden = self.transformer(hidden)               # [B, L, D]
        logits = self.output_projection(hidden)         # [B, L, 256]

        return logits

    def _get_loss(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_start: [B, 4097]  clean sequence with class token prepended
            t:        [B]       timesteps
        Returns:
            scalar loss
        """
        x_noisy = self.noise_schedule.q_sample(x_start, t)  # [B, 4097]
        logits  = self(x_noisy, t)                           # [B, 4097, 256]

        # Only compute loss at MASKED positions (and skip position 0 = class token)
        mask = (x_noisy == self.mask_id)   # [B, 4097]
        mask[:, 0] = False                 # never predict class token position

        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_start.device, requires_grad=True)

        pred    = logits.reshape(-1, NUM_MESH_CODES)[mask.reshape(-1)]   # [M, 256]
        targets = x_start.reshape(-1)[mask.reshape(-1)]                  # [M]

        return F.cross_entropy(pred, targets)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x = batch['input_ids']  # [B, 4097]
        t = torch.randint(0, self.noise_schedule.num_timesteps, (x.shape[0],), device=x.device)
        loss = self._get_loss(x, t)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x = batch['input_ids']  # [B, 4097]
        t = torch.randint(0, self.noise_schedule.num_timesteps, (x.shape[0],), device=x.device)
        loss = self._get_loss(x, t)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )

    @torch.no_grad()
    def generate(
        self,
        class_ids: torch.Tensor,
        temperature: float = 1.0,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Generate mesh token sequences conditioned on class tokens.

        Args:
            class_ids: [B]   — class indices (0-39)
            temperature: float
            num_steps: int   — number of denoising steps

        Returns:
            sequences: [B, 4097]  — full sequence with class token at position 0
        """
        device = next(self.parameters()).device
        B = class_ids.shape[0]

        # Build initial sequence: [class_token, MASK, MASK, ..., MASK]
        class_tokens = class_to_token(class_ids).to(device)          # [B]
        x = torch.full((B, self.seq_len), self.mask_id, dtype=torch.long, device=device)
        x[:, 0] = class_tokens                                        # fix class token

        # Denoising steps: high t → low t
        step_size = self.noise_schedule.num_timesteps // num_steps
        timesteps = list(range(self.noise_schedule.num_timesteps - 1, -1, -step_size))

        for t_val in timesteps:
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
            logits   = self(x, t_tensor)             # [B, 4097, 256]

            if temperature > 0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)        # [B, 4097, 256]

            # Only unmask the mesh positions (1:)
            mask = (x == self.mask_id)
            mask[:, 0] = False                        # never touch class token

            if mask.any():
                # Sample from predicted distribution for masked positions
                flat_logits  = logits[mask]                              # [M, 256]
                flat_samples = torch.multinomial(F.softmax(flat_logits / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
                x_new = x.clone()
                x_new[mask] = flat_samples
                x = x_new

        return x  # [B, 4097]
