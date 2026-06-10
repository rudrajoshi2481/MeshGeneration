import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union


class DiscreteNoiseSchedule(nn.Module):
    """
    Discrete diffusion noise schedule that gradually replaces tokens with mask tokens.
    """
    def __init__(
        self, 
        num_timesteps: int, 
        vocab_size: int, 
        mask_id: int,
        schedule_type: str = 'linear'
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        
        # Create noise schedule
        if schedule_type == 'linear':
            # Linear beta schedule
            self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule as in improved DDPM
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Calculate alphas and derived values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For convenience in sampling
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Original sequence of tokens [B, L]
            t: Timesteps [B]
            noise: Optional pre-generated noise
            
        Returns:
            Noisy tokens at timestep t
        """
        # Get noise level for each sample in batch
        noise_level = self.alphas_cumprod.to(x_start.device)[t]
        
        # Create a random mask that determines which positions to corrupt
        mask_shape = (x_start.shape[0], x_start.shape[1])
        
        # Generate random noise or use provided noise
        if noise is None:
            noise = torch.rand(mask_shape, device=x_start.device)
        
        # For each position, mask if random value < noise level
        # This makes earlier timesteps have fewer masks (low noise_level),
        # and later timesteps have more masks (high noise_level)
        mask = (noise >= noise_level.unsqueeze(-1)).to(x_start.device)
        
        # Replace masked positions with the mask token
        x_noisy = x_start.clone()
        x_noisy[mask] = self.mask_id
        
        return x_noisy
    
    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the posterior distribution q(x_{t-1} | x_t, x_0)
        
        Args:
            x_0: Original sequence [B, L]
            x_t: Noisy sequence at timestep t [B, L]
            t: Timestep [B]
            
        Returns:
            Distribution for x_{t-1}
        """
        # This is a simplified posterior for discrete tokens
        # We'll use this for the DDIM-style sampler
        posterior_mean = self._posterior_mean(x_0, x_t, t)
        return posterior_mean
    
    def _posterior_mean(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Helper function for posterior mean calculation"""
        # For discrete tokens, we calculate a probability of keeping original tokens vs noise
        posterior_mean = (
            self.alphas_cumprod_prev[t].unsqueeze(-1) * x_0 +
            (1 - self.alphas_cumprod_prev[t]).unsqueeze(-1) * x_t
        )
        return posterior_mean


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class TimeEmbedding(nn.Module):
    """
    Embeds diffusion timestep into a vector of dimension d_model.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create an embedding for timestep t
        
        Args:
            t: Timesteps [B]
            
        Returns:
            Timestep embeddings [B, D]
        """
        # Normalize t to [0, 1]
        t_norm = t.float() / 1000.0
        
        # Create embedding
        t_embed = self.time_embed(t_norm.unsqueeze(-1))
        
        return t_embed


class ClassEmbedding(nn.Module):
    """
    Embeds class labels into a vector of dimension d_model.
    """
    def __init__(self, num_classes: int, d_model: int):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, d_model)
        
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Create an embedding for class label c
        
        Args:
            c: Class labels [B]
            
        Returns:
            Class embeddings [B, D]
        """
        return self.class_embed(c)


class DiscreteDiffusionTransformer(pl.LightningModule):
    """
    Transformer-based diffusion model for discrete token sequences.
    Predicts the original tokens from noisy tokens at various timesteps.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        mask_id: int = None,
        num_timesteps: int = 1000,
        schedule_type: str = 'linear',
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.mask_id = mask_id if mask_id is not None else vocab_size  # Default to vocab_size if not specified
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Create noise schedule
        self.noise_schedule = DiscreteNoiseSchedule(
            num_timesteps=num_timesteps,
            vocab_size=vocab_size,
            mask_id=self.mask_id,
            schedule_type=schedule_type
        )
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(d_model)
        
        # Class conditioning (optional)
        self.use_class_condition = num_classes is not None
        if self.use_class_condition:
            self.class_embedding = ClassEmbedding(num_classes, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection to vocab
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion transformer
        
        Args:
            x: Input token sequence (possibly masked) [B, L]
            timesteps: Diffusion timesteps for each sequence [B]
            class_labels: Optional class labels for conditioning [B]
            
        Returns:
            Logits for token prediction [B, L, V]
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Get token embeddings
        token_embed = self.token_embedding(x)  # [B, L, D]
        
        # Add positional encoding
        token_embed = self.pos_encoder(token_embed)  # [B, L, D]
        
        # Get time embeddings and expand to match sequence length
        time_embed = self.time_embedding(timesteps)  # [B, D]
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]
        
        # Add time embedding to token embedding
        hidden_states = token_embed + time_embed  # [B, L, D]
        
        # Add class conditioning if provided
        if self.use_class_condition and class_labels is not None:
            class_embed = self.class_embedding(class_labels)  # [B, D]
            class_embed = class_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]
            hidden_states = hidden_states + class_embed
        
        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(hidden_states)  # [B, L, D]
        
        # Project to vocabulary
        logits = self.output_projection(encoder_output)  # [B, L, V]
        
        return logits
    
    def _get_loss(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate the diffusion loss for a batch
        
        Args:
            x_start: Original token sequence [B, L]
            t: Timesteps [B]
            class_labels: Optional class labels [B]
            
        Returns:
            Loss value
        """
        # Add noise to the input sequence
        x_noisy = self.noise_schedule.q_sample(x_start, t)
        
        # Predict the original sequence
        pred_logits = self(x_noisy, t, class_labels)
        
        # Calculate cross-entropy loss
        # We only want to calculate loss for masked positions
        mask = (x_noisy == self.mask_id)
        
        if mask.sum() == 0:
            # If there are no masked positions, return zero loss
            return torch.tensor(0.0, device=x_start.device, requires_grad=True)
        
        # Reshape for loss calculation
        pred_logits = pred_logits.reshape(-1, self.vocab_size)[mask.reshape(-1)]
        targets = x_start.reshape(-1)[mask.reshape(-1)]
        
        loss = F.cross_entropy(pred_logits, targets)
        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for Lightning
        
        Args:
            batch: Batch of data containing 'input_ids' and optionally 'labels'
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x = batch['input_ids']
        class_labels = batch['class_labels'] if 'class_labels' in batch and self.use_class_condition else None
        
        # Sample random timesteps
        t = torch.randint(0, self.noise_schedule.num_timesteps, (x.shape[0],), device=x.device)
        
        # Calculate loss
        loss = self._get_loss(x, t, class_labels)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for Lightning
        
        Args:
            batch: Batch of data containing 'input_ids' and optionally 'labels'
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x = batch['input_ids']
        class_labels = batch['class_labels'] if 'class_labels' in batch and self.use_class_condition else None
        
        # Sample random timesteps
        t = torch.randint(0, self.noise_schedule.num_timesteps, (x.shape[0],), device=x.device)
        
        # Calculate loss
        loss = self._get_loss(x, t, class_labels)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for Lightning
        
        Returns:
            Optimizer
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay
        )
        
        return optimizer
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int = 1, 
        seq_len: Optional[int] = None,
        class_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        num_steps: Optional[int] = None,
        ddim_sampling_eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate samples from the diffusion model using DDIM sampling
        
        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate (defaults to max_seq_len)
            class_labels: Optional class labels for conditional generation
            temperature: Sampling temperature (higher = more diverse)
            num_steps: Number of sampling steps (defaults to num_timesteps)
            ddim_sampling_eta: Parameter between 0 and 1 controlling stochasticity
            
        Returns:
            Generated token sequences [B, L]
        """
        device = next(self.parameters()).device
        seq_len = seq_len or self.max_seq_len
        num_steps = num_steps or self.noise_schedule.num_timesteps
        
        # Start from completely masked sequence
        x = torch.full((batch_size, seq_len), self.mask_id, dtype=torch.long, device=device)
        
        # Time schedule for sampling
        timesteps = list(range(self.noise_schedule.num_timesteps - 1, -1, -self.noise_schedule.num_timesteps // num_steps))
        
        # DDIM sampling
        for i, t in enumerate(timesteps):
            # Tensor of timesteps for the batch
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction (logits)
            logits = self(x, t_tensor, class_labels)
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            # We only sample for masked positions
            mask = (x == self.mask_id)
            
            if mask.any():
                # Sample from predicted distributions for masked positions
                masked_logits = logits[mask]
                masked_probs = F.softmax(masked_logits, dim=-1)
                masked_samples = torch.multinomial(masked_probs, 1).squeeze(-1)
                
                # Update x
                x_new = x.clone()
                x_new[mask] = masked_samples
                x = x_new
        
        return x
    
    @torch.no_grad()
    def generate(
        self, 
        batch_size: int = 1, 
        seq_len: Optional[int] = None,
        class_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate samples from the model (alias for sample)
        
        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate (defaults to max_seq_len)
            class_labels: Optional class labels for conditional generation
            temperature: Sampling temperature (higher = more diverse)
            num_steps: Number of sampling steps (defaults to 50)
            
        Returns:
            Generated token sequences [B, L]
        """
        return self.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            class_labels=class_labels,
            temperature=temperature,
            num_steps=num_steps
        )