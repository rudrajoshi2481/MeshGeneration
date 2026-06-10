# MeshGPT VQ-VAE

3D mesh autoencoder using Vector Quantization for discrete latent representation.

## Quick Run

```bash
# Train without classifier (recommended for semantic channel)
cd training_scripts
python train_without_classifier.py --quick        # 1 epoch test
python train_without_classifier.py --gpus 1     # Full training

# Extract codes for downstream models
cd ../models/diffusion
python extract_fresh_codes.py \
    --ckpt <path/to/checkpoint> \
    --out_dir ../../trash/data
```

## Config Changes

Edit `src/config.py` lines 110-140:

```python
class SmallModelConfig:
    vq: VQConfig = field(default_factory=lambda: VQConfig(
        num_embeddings=256,    # Codebook size
        embedding_dim=64,      # Token dimension
    ))
    train: TrainConfig = field(default_factory=lambda: TrainConfig(
        batch_size=4,          # GPU memory dependent
        max_steps=100_000,
        lr=1e-3
    ))
```

## Args

- `--quick` - 2 epoch smoke test (32 samples)
- `--gpus` - Number of GPUs (default: 1)
- `--resume` - Resume from checkpoint (`last` or path)

## Output

- Checkpoints: `runs/<run_name>/checkpoints/`
- Logs: CSVLogger + plots/
- Codes: `trash/data/train_codes.pt` + `val_codes.pt`

## Architecture

- **Encoder**: PointNet-style encoder with grid-based feature aggregation (16³ voxel grid)
- **Masker**: Geometric masking for adaptive resolution
- **VQ Codebook**: 256 entries, 64-dimensional embeddings with EMA updates
- **Decoder**: Occupancy prediction from quantized codes
- **Classifier** (optional): Shape classification head
- **Fingerprint**: Geometric feature extraction

## Model Variants

### Small Model (1.44M params)
- Grid resolution: 16³
- Codebook: 256 entries × 64 dims
- Encoder: 919K params
- Decoder: 74K params

### Large Model
- Higher capacity for complex shapes
- Configurable in `src/config.py`

## Directory Structure

```
mesh_vqvae/
├── src/
│   ├── config.py          # Model configurations
│   ├── model.py           # Main VQ-VAE model
│   ├── encoder.py         # Point cloud encoder
│   ├── decoder.py         # Occupancy decoder
│   ├── quantizer.py       # Vector quantization with EMA
│   ├── masker.py          # Geometric masking
│   ├── heads.py           # Classifier & fingerprint heads
│   ├── dataset.py         # ModelNet40 dataset loader
│   ├── preprocessing.py   # Mesh preprocessing utilities
│   ├── train.py           # Original training script
│   ├── evaluate.py        # Evaluation & visualization
│   └── stage3_decode.py   # Decoding from latent codes
├── training_scripts/
│   ├── train_with_classifier.py    # Full model (1.91M params)
│   └── train_without_classifier.py # Pure VQ-VAE (1.44M params)
```

## Training

### With Classifier (Multi-task)
```bash
cd training_scripts
python train_with_classifier.py --gpus 8 --model small
```

### Without Classifier (Pure Reconstruction)
```bash
cd training_scripts
python train_without_classifier.py --gpus 8 --model small
```

### Quick Smoke Test
```bash
python train_without_classifier.py --quick
```

## Features

- **Distributed Training**: 8-GPU support with DDP
- **Mixed Precision**: bf16 for efficiency
- **Early Stopping**: Automatic convergence detection
- **Comprehensive Plotting**: 
  - Reconstruction visualizations
  - IoU distributions
  - Codebook utilization
  - t-SNE embeddings
  - Per-class metrics

## Performance

- **Target IoU**: 0.38-0.42 (ModelNet40)
- **Codebook Utilization**: >80%
- **Training Time**: ~2-3 hours on 8×A100

## Extract Codes for Diffusion

```bash
python src/extract_fresh_codes.py --ckpt <path/to/checkpoint.ckpt> --out_dir ../../trash/data
```

Outputs:
- `train_codes.pt`: [N, 4096] discrete codes
- `val_codes.pt`: [M, 4096] discrete codes

## Requirements

- PyTorch 2.0+
- PyTorch Lightning
- trimesh
- matplotlib
- numpy
