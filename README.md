**NOTE: avoid src forlder it is previous implementation**
# MeshGPT: 3D Shape Generation via VQ-VAE and Discrete Diffusion

Two-stage generative model for 3D shapes using Vector Quantization and Score-matching Discrete Diffusion.

## Overview

This repository contains:
1. **MeshGPT VQ-VAE** (`mesh_vqvae/`) - Autoencoder for 3D meshes with discrete latent codes
2. **SEDD Diffusion** (`diffusion_model/`) - Discrete diffusion model for generating latent codes

## Explanation

```sh
╔═════════════════════════════════════════════════════════════════════════════════╗
║                           STAGE 1: MESHGPT VQ-VAE TRAINING                      ║
╚═════════════════════════════════════════════════════════════════════════════════╝

3D Mesh (ModelNet40)                    MeshGPT VQ-VAE                    Reconstructed Mesh
┌─────────────────┐                      ┌─────────────────┐                ┌─────────────────┐
│  • Points       │ ──► Encode ───────►  │  Encoder        │──► Quantize ──►│  Decoder        │ ──► Output
│  • Normals      │                      │  PointNet-style │                │  Occupancy      │
│  • Curvature    │                      │  → [4096, 64]   │                │  Prediction     │
│  • Query Pts    │                      │                 │                │                 │
│  • Occupancy    │                      │                 │                │                 │
└─────────────────┘                      └─────────────────┘                └─────────────────┘
       │                                         │                                 │
       │                                         ▼                                 ▼
       │                                 ┌─────────────────┐                ┌─────────────────┐
       │                                 │ Vector Quantizer│                │  IoU: 0.38-0.42 │
       │                                 │ 256 entries     │                │  Loss: recon+vq │
       │                                 │ [64-dim each]   │                │                 │
       │                                 └─────────────────┘                └─────────────────┘
       │                                         │
       │                                         ▼
       │                                 ┌─────────────────┐
       └────────────────────────────────►│ Codebook Indices│
                                         │ [4096] tokens   │
                                         │ Values: 0-255   │
                                         └─────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════╗
║                           STAGE 2: SEDD DIFFUSION TRAINING                      ║
╚═════════════════════════════════════════════════════════════════════════════════╝

Codebook Indices                    SEDD Transformer                      Generated Indices
┌─────────────────┐                  ┌─────────────────┐                  ┌─────────────────┐
│ Real Codes      │ ──► Add Noise──► │  Transformer    │ ──► Predict ───► │  Predicted      │
│ [B, 4096]       │                  │  d_model=512    │                  │  Codes          │
│ Values: 0-255   │                  │  num_layers=6   │                  │  [B, 4096]      │
│                 │                  │  nhead=8        │                  │  Values: 0-255  │
└─────────────────┘                  └─────────────────┘                  └─────────────────┘
       │                                         │                                 │
       │                                         ▼                                 ▼
       │                                 ┌─────────────────┐                ┌─────────────────┐
       │                                 │ Embeddings:     │                │  Overlap: 90%   │
       │                                 │ • Token:[257,512]│               │  Val Loss: 1.9  │
       │                                 │ • Position      │                │                 │
       │                                 │ • Time          │                │                 │
       │                                 │ • Class (opt)   │                │                 │
       │                                 └─────────────────┘                └─────────────────┘
       │                                         │
       │                                         ▼
       │                                 ┌─────────────────┐
       └───────────────────────────────► │ Diffusion Process│
                                         │ t=0: clean      │
                                         │ t=500: 50% mask │
                                         │ t=1000: all mask│
                                         └─────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════╗
║                           STAGE 3: GENERATION & DECODING                        ║
╚═════════════════════════════════════════════════════════════════════════════════╝

Generated Codes                      MeshGPT Decoder                     Final 3D Mesh
┌─────────────────┐                  ┌─────────────────┐                  ┌─────────────────┐
│ SEDD Output     │ ──► Lookup ────► │  Decoder        │ ──► Reconstruct ─► │  Generated    │
│ [4096] tokens   │                  │  Pre-trained    │                  │  3D Shape       │
│ Values: 0-255   │                  │  from Stage 1   │                  │  • Occupancy    │
│                 │                  │                 │                  │  • Geometry     │
└─────────────────┘                  └─────────────────┘                  └─────────────────┘
       │                                         │                                 │
       │                                         ▼                                 ▼
       │                                 ┌─────────────────┐                ┌─────────────────┐
       │                                 │ Codebook Lookup │                │  New Mesh       │
       │                                 │ Get 64-dim      │                │  Never seen     │
       │                                 │ embeddings      │                │  before!        │
       │                                 └─────────────────┘                └─────────────────┘
       │                                         │
       │                                         ▼
       │                                 ┌─────────────────┐
       └────────────────────────────────►│ Continuous      │
                                         │ Features        │
                                         │ [4096, 64]      │
                                         └─────────────────┘
```
## Quick Start

### 1. Train MeshGPT VQ-VAE

```bash
cd mesh_vqvae/training_scripts

# Option A: With classifier (multi-task learning)
python train_with_classifier.py --gpus 8 --model small

# Option B: Without classifier (pure reconstruction)
python train_without_classifier.py --gpus 8 --model small

# Quick test (2 epochs, 1 GPU)
python train_without_classifier.py --quick
```

**Outputs**: Checkpoints, plots, and metrics in configured output directory

### 2. Extract Discrete Codes

```bash
cd mesh_vqvae
python extract_codes.py \
    --ckpt path/to/best_checkpoint.ckpt \
    --out_dir ./codes
```

**Outputs**: 
- `codes/train_codes.pt` - [2720, 4096] discrete codes
- `codes/val_codes.pt` - [480, 4096] discrete codes

### 3. Train SEDD Diffusion Model

```bash
cd ../diffusion_model

# Small test (200 samples)
python train_sedd.py --mode small

# Full training (8 GPUs)
python train_sedd.py --mode full
```

**Outputs**: Checkpoints, training curves, generation samples

### 4. Generate New Shapes

```python
# Load trained models
from mesh_vqvae.src.model import MaskedVQVAE3D
from diffusion_model.SEDD import DiscreteDiffusionTransformer

# Load SEDD
sedd = DiscreteDiffusionTransformer.load_from_checkpoint(sedd_ckpt)

# Generate codes
codes = sedd.sample(batch_size=10, num_steps=50)  # [10, 4096]

# Load MeshGPT decoder
meshgpt = MaskedVQVAE3D.load_from_checkpoint(meshgpt_ckpt)

# Decode to 3D shapes
shapes = meshgpt.decode_from_codes(codes)
```

## Repository Structure

```
push_github_src/
├── mesh_vqvae/              # VQ-VAE for 3D meshes
│   ├── src/                 # Core model code
│   │   ├── model.py         # Main VQ-VAE
│   │   ├── encoder.py       # Point cloud encoder
│   │   ├── decoder.py       # Occupancy decoder
│   │   ├── quantizer.py     # Vector quantization
│   │   ├── config.py        # Model configs
│   │   ├── dataset.py       # ModelNet40 loader
│   │   └── ...
│   ├── training_scripts/
│   │   ├── train_with_classifier.py
│   │   └── train_without_classifier.py
│   ├── extract_codes.py     # Extract discrete codes
│   └── README.md
│
├── diffusion_model/         # SEDD for code generation
│   ├── SEDD.py              # Discrete diffusion transformer
│   ├── train_sedd.py        # Training script
│   ├── preprocessing.py     # Utilities
│   └── README.md
│
└── README.md                # This file
```

## Model Performance

### MeshGPT VQ-VAE
- **IoU**: 0.38-0.42 (ModelNet40)
- **Codebook Utilization**: >80%
- **Model Size**: 1.44M params (without classifier), 1.91M (with classifier)
- **Training Time**: 2-3 hours on 8×A100

### SEDD Diffusion
- **Validation Loss**: 1.917
- **Code Overlap**: 90.19% (real vs generated)
- **Model Size**: 19.5M params (full config)
- **Training Time**: ~40 minutes on 8×A100

## Features

### MeshGPT VQ-VAE
- ✅ Discrete latent representation (256 codes)
- ✅ Optional classifier head for multi-task learning
- ✅ Geometric fingerprint extraction
- ✅ Comprehensive visualization (IoU, t-SNE, reconstruction)
- ✅ Distributed training (DDP, bf16)
- ✅ Early stopping and checkpointing

### SEDD Diffusion
- ✅ Class-conditional generation
- ✅ Score-matching for discrete distributions
- ✅ Transformer architecture
- ✅ Training curve visualization
- ✅ Real vs generated distribution analysis

## Requirements

Both modules require:
- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- CUDA-capable GPU(s)

See individual `requirements.txt` files in each module.

## Dataset

Models are trained on **ModelNet40**:
- 40 object categories
- 2720 training samples
- 480 validation samples
- Point cloud + occupancy representation
