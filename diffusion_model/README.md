# SEDD: Score-matching for Discrete Diffusion

Discrete diffusion model for generating MeshGPT latent code sequences.

## Overview

SEDD learns to generate discrete code sequences (extracted from MeshGPT VQ-VAE) using score-matching diffusion. This enables unconditional and class-conditional 3D shape generation in the latent space.

## Architecture

- **Transformer-based**: Multi-head attention over code sequences
- **Discrete Diffusion**: Score-matching for categorical distributions
- **Class-Conditional**: Optional class conditioning for controlled generation
- **Sequence Length**: 4096 codes (16³ voxel grid)
- **Vocabulary**: 256 discrete codes

## Model Configurations

### Small (Quick Test)
- d_model: 256
- num_layers: 4
- nhead: 4
- Parameters: ~5M

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
diffusion_model/
├── SEDD.py              # Core SEDD implementation
├── train_sedd.py        # Training script with plotting
├── preprocessing.py     # ModelNet40 class names (for plotting)
└── README.md
```

## Training Pipeline

### Step 1: Extract MeshGPT Codes
First, train MeshGPT and extract discrete codes:
```bash
cd ../mesh_vqvae
python extract_codes.py --ckpt path/to/meshgpt.ckpt --out_dir ./codes
```

This generates:
- `train_codes.pt`: [2720, 4096] codes
- `val_codes.pt`: [480, 4096] codes

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
