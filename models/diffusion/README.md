# SEDD: Score-matching for Discrete Diffusion

Discrete diffusion model for generating MeshGPT latent code sequences.

## Quick Start

```bash
# Test (1 epoch, small model, conditional)
python train_sedd.py --model_mode small --condition_mode conditional --epochs 1

# Full training (small model, conditional) - RECOMMENDED
python train_sedd.py --model_mode small --condition_mode conditional --epochs 200

# Unconditional training (optional comparison)
python train_sedd.py --model_mode small --condition_mode unconditional --epochs 200
```

## Required Data

`--data_dir` must contain:
- `train_codes.pt` - Training tokens [N, 4096] with labels
- `val_codes.pt` - Validation tokens [M, 4096] with labels

Default: `trash/data/`

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_mode` | `small` | Model size: small=128d, medium=256d, full=512d |
| `--condition_mode` | `conditional` | Use class labels: conditional / unconditional |
| `--epochs` | 200 | Training epochs |
| `--batch_size` | 16 | Batch size per GPU (reduce if OOM) |
| `--gpus` | 1 | Number of GPUs |
| `--n_train` | -1 | Training samples (-1=all) |
| `--n_val` | -1 | Validation samples (-1=all) |
| `--data_dir` | `trash/data` | Path to train_codes.pt and val_codes.pt |
| `--out_base` | `trash/sedd_runs` | Output directory base |

## Outputs

```
trash/sedd_runs/sedd_<model_mode>_<condition_mode>_<timestamp>/
├── checkpoints/
│   ├── sedd_...-best.ckpt    ← Best checkpoint
│   └── sedd_...-last.ckpt    ← Last checkpoint
├── plots/
│   ├── curves_epXXXX.png     ← Training curves + perplexity
│   ├── code_dist_epXXXX.png  ← Real vs generated code distribution
│   └── gen_hist_epXXXX.png   ← Generated histograms per class
└── report.json               ← Training summary
```

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
├── train_sedd.py        # Training script (conditional + unconditional)
├── extract_fresh_codes.py  # Extract codes from VQVAE
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

**Quick test (1 epoch)**:
```bash
python train_sedd.py --model_mode small --condition_mode conditional --epochs 1
```

**Full training (recommended)**:
```bash
python train_sedd.py \
    --model_mode small \
    --condition_mode conditional \
    --epochs 200 \
    --n_train -1 \
    --batch_size 16 \
    --gpus 1
```

**Custom paths**:
```bash
python train_sedd.py \
    --data_dir /path/to/data \
    --out_base /path/to/output \
    --model_mode small \
    --condition_mode conditional \
    --epochs 200
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
- **Conditional mode**: Uses class labels for controlled generation
- **Unconditional mode**: Generates without class labels
- Diffusion process: 1000 timesteps with cosine schedule

## GPU Memory Guide

| Model | VRAM | Use When |
|-------|------|----------|
| small (128d) | ~4GB | Your GPU (~11GB), testing |
| medium (256d) | ~8GB | Balanced speed/quality |
| full (512d) | ~20GB | Best quality, production |
