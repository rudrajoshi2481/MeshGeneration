"""
config.py — All hyperparameters in one place.
Small model: fast iteration, overfit tests.
Large model: full training on 8x A100.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EncoderConfig:
    in_channels: int = 7          # XYZ (3) + normals (3) + curvature (1)
    patch_size: int = 32          # points per patch
    num_patches: int = 64         # FPS anchor points
    patch_dim: int = 128          # per-patch embedding dim (small) / 256 (large)
    grid_res: int = 16            # voxel grid resolution (small) / 32 (large)
    grid_channels: int = 128      # grid feature channels (small) / 256 (large)
    conv_layers: int = 2          # 3D conv downsampler depth


@dataclass
class MaskerConfig:
    input_dim: int = 128          # must match EncoderConfig.grid_channels
    codebook_dim: int = 32        # projected dim before VQ
    topk_ratio: float = 0.75      # fraction of voxels KEPT
    num_classes: int = 40         # ModelNet40


@dataclass
class VQConfig:
    num_embeddings: int = 512     # codebook size (small) / 4096 (large)
    embedding_dim: int = 32       # codebook vector dim
    commitment_weight: float = 1.0
    ema_decay: float = 0.97           # faster adaptation (was 0.99)
    threshold_ema_dead_code: float = 1.0  # restart codes used < 1 time (was 2.0)
    use_rotation_trick: bool = False  # swap STE for rotation trick


@dataclass
class DemaskerConfig:
    input_dim: int = 32           # codebook_dim
    output_dim: int = 128         # must match grid_channels
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    grid_channels: int = 128      # must match grid_channels
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.0


@dataclass
class ClassifierConfig:
    codebook_size: int = 512      # must match VQConfig.num_embeddings
    num_classes: int = 40
    hidden_dim: int = 256


@dataclass
class FingerprintConfig:
    codebook_dim: int = 32        # must match VQConfig.embedding_dim
    fingerprint_dim: int = 128
    temperature: float = 0.07


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "/data/joshi/modelnet40_meshes"
    cache_dir: str = "/data/joshi/tmp/MeshGeneration/runs/cache"
    output_dir: str = "/data/joshi/tmp/MeshGeneration/runs"
    num_workers: int = 8
    num_surface_points: int = 2048
    num_query_points: int = 2048

    # Training
    batch_size: int = 16          # per GPU (small) → 32 (large)
    num_gpus: int = 8
    max_steps: int = 100_000
    lr: float = 5e-4              # Reduced from 1e-3 for stability
    classifier_lr: float = 2e-3   # Higher LR for classifier (needs stronger signal)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 500       # Faster warmup

    # Loss weights
    vq_beta: float = 1.0
    cls_weight_max: float = 2.0   # Increased to prioritize classification
    cls_ramp_steps: int = 200     # Very fast ramp (~9 epochs)
    fp_weight_max: float = 0.15
    fp_ramp_steps: int = 400      # ~19 epochs

    # Model options
    use_classifier: bool = True   # Set to False to disable classifier head

    # Logging
    log_every: int = 100
    val_every: int = 2000
    save_every: int = 5000

    # Misc
    seed: int = 42
    precision: str = "bf16-mixed"  # bfloat16 on A100


@dataclass
class SmallModelConfig:
    """Tiny model for fast overfit/test iteration."""
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        patch_dim=128, grid_res=16, grid_channels=128, conv_layers=2
    ))
    masker: MaskerConfig = field(default_factory=lambda: MaskerConfig(
        input_dim=128, codebook_dim=64, topk_ratio=0.5, num_classes=40
    ))
    vq: VQConfig = field(default_factory=lambda: VQConfig(
        num_embeddings=256, embedding_dim=64,
        ema_decay=0.97, threshold_ema_dead_code=1.0
    ))
    demasker: DemaskerConfig = field(default_factory=lambda: DemaskerConfig(
        input_dim=64, output_dim=128, num_heads=4, num_layers=3
    ))
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig(
        grid_channels=128, hidden_dims=[256, 128, 64]
    ))
    classifier: ClassifierConfig = field(default_factory=lambda: ClassifierConfig(
        codebook_size=256, num_classes=40, hidden_dim=256  # Doubled capacity
    ))
    fingerprint: FingerprintConfig = field(default_factory=lambda: FingerprintConfig(
        codebook_dim=64, fingerprint_dim=128
    ))
    train: TrainConfig = field(default_factory=lambda: TrainConfig(
        batch_size=16, max_steps=100_000, lr=1e-3
    ))
    num_classes: int = 40
    fingerprint_dim: int = 128


@dataclass
class LargeModelConfig:
    """Full model for 8x A100 training."""
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        in_channels=7, patch_size=32, num_patches=128,
        patch_dim=256, grid_res=32, grid_channels=256, conv_layers=3
    ))
    masker: MaskerConfig = field(default_factory=lambda: MaskerConfig(
        input_dim=256, codebook_dim=32, topk_ratio=0.75, num_classes=40
    ))
    vq: VQConfig = field(default_factory=lambda: VQConfig(
        num_embeddings=4096, embedding_dim=32
    ))
    demasker: DemaskerConfig = field(default_factory=lambda: DemaskerConfig(
        input_dim=32, output_dim=256, num_heads=8, num_layers=4
    ))
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig(
        grid_channels=256, hidden_dims=[512, 256, 128]
    ))
    classifier: ClassifierConfig = field(default_factory=lambda: ClassifierConfig(
        codebook_size=4096, num_classes=40, hidden_dim=512
    ))
    fingerprint: FingerprintConfig = field(default_factory=lambda: FingerprintConfig(
        codebook_dim=32, fingerprint_dim=256
    ))
    train: TrainConfig = field(default_factory=lambda: TrainConfig(
        batch_size=32, max_steps=200_000,
        cls_ramp_steps=5000, fp_ramp_steps=10000
    ))
    num_classes: int = 40
    fingerprint_dim: int = 256
