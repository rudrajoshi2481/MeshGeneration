# SEDD: Score-matching for Discrete Diffusion

Discrete diffusion model for generating MeshGPT latent code sequences.

## Quick Run

```bash
# Train SEDD (small config, 1 GPU)
python train_sedd.py --mode small --epochs 200 --gpus 1 --batch_size 16

# Quick test (1 epoch)
python train_sedd.py --mode small --epochs 1 --gpus 1 --batch_size 16
```

## Config Changes

Edit `train_sedd.py` lines 25-50:

```python
SMALL_CFG = {
    "vocab_size": 256,
    "max_seq_len": 4096,
    "d_model": 128,      # Model dimension
    "nhead": 4,          # Attention heads
    "num_layers": 3,     # Transformer layers
    "num_classes": 40,
    "learning_rate": 1e-4
}
```

## Args

- `--mode` - `small` (128d) / `medium` (256d) / `full` (512d)
- `--epochs` - Training epochs (default: 200)
- `--gpus` - Number of GPUs (default: 1)
- `--batch_size` - Batch size (default: 16)
- `--n_train` - Samples to use (-1 for all)

## Output

- Checkpoints: `trash/sedd_runs/<run_id>/checkpoints/`
- Plots: Loss curves, generation histograms

## Overview

SEDD learns to generate discrete code sequences (extracted from MeshGPT VQ-VAE) using score-matching diffusion. This enables unconditional and class-conditional 3D shape generation in the latent space.

## Architecture

- **Transformer-based**: Multi-head attention over code sequences
- **Discrete Diffusion**: Score-matching for categorical distributions
- **Class-Conditional**: Optional class conditioning for controlled generation
- **Sequence Length**: 4096 codes (16³ voxel grid)
- **Vocabulary**: 256 discrete codes

## Model Configurations

### Small (Your Config)
- d_model: 128
- num_layers: 3
- nhead: 4
- Parameters: ~1.8M

### Medium
- d_model: 384
- num_layers: 6
- nhead: 8
- Parameters: ~12M

### Full
- d_model: 512
- num_layers: 6
- nhead: 8
- Parameters: ~19M

## Directory Structure

```
models/diffusion/
├── SEDD.py              # Core SEDD implementation
├── train_sedd.py        # Training script with plotting
├── extract_fresh_codes.py  # Extract codes from VQVAE
├── preprocessing.py     # ModelNet40 class names
└── README.md
```

## Training Pipeline

### Step 1: Extract MeshGPT Codes
First, train MeshGPT and extract discrete codes:
```bash
# From models/diffusion/ directory (where this README is)
python extract_fresh_codes.py \
    --ckpt <path/to/vqvae_checkpoint.ckpt> \
    --out_dir ../../trash/data
```

This generates:
- `train_codes.pt`: [~6400, 4096] codes (depends on split)
- `val_codes.pt`: [~1600, 4096] codes

### Step 2: Train SEDD

**Small test (200 samples, 30 epochs)**:
```bash
python train_sedd.py --mode small
```

**Medium test (1000 samples, 50 epochs)**:
```bash
python train_sedd.py --mode medium
```

**Full training (all samples, 200 epochs, 8 GPUs)**:
```bash
python train_sedd.py --mode full
```

## Outputs

Training generates:
```
sedd_runs/<run_id>/
├── checkpoints/
│   └── sedd-epoch=XXXX-val_loss=X.XXXX.ckpt
├── plots/
│   ├── curves_epXXXX.png           # Training curves
│   ├── gen_hist_epXXXX.png         # Generated code histograms
│   └── code_dist_epXXXX.png        # Real vs generated distribution
├── lightning_logs/
└── report.json                      # Final metrics
```

## Key Metrics

- **Validation Loss**: Lower is better (~1.9 for full model)
- **Code Overlap**: Percentage of real codes present in generated samples (~90%)
- **Distribution Match**: Visual comparison of real vs generated code usage

## Performance (Full Model)

- **Best Val Loss**: 1.917
- **Code Overlap**: 90.19%
- **Training Time**: ~40 minutes on 8×A100
- **Epochs**: 80 (with early stopping)

## Generation

After training, use the best checkpoint to generate new shapes:
```python
from SEDD import DiscreteDiffusionTransformer

model = DiscreteDiffusionTransformer.load_from_checkpoint(ckpt_path)
samples = model.sample(
    batch_size=10,
    class_labels=torch.arange(10),  # One per class
    num_steps=50
)  # [10, 4096] discrete codes
```

Then decode with MeshGPT decoder to get 3D shapes.

## Requirements

- PyTorch 2.0+
- PyTorch Lightning
- numpy
- matplotlib

## Integration with MeshGPT

1. Train MeshGPT VQ-VAE (see `../mesh_vqvae/`)
2. Extract codes using `extract_codes.py`
3. Train SEDD on extracted codes
4. Generate new code sequences
5. Decode with MeshGPT decoder to get 3D meshes

## Notes

- SEDD operates in the discrete latent space (256 codes)
- Each shape is represented as 4096 code indices
- Class conditioning enables controlled generation
- Diffusion process: 1000 timesteps with cosine schedule
