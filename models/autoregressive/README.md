# DoT (Decoder-only Transformer)

Autoregressive model for token sequence generation (NanoGPT architecture).

## Quick Start

```bash
# Quick test (1 epoch, small model, conditional)
python train_dot_mesh.py --model_mode small --condition_mode conditional --epochs 1

# Full training (small model, conditional - RECOMMENDED)
python train_dot_mesh.py --model_mode small --condition_mode conditional --epochs 100

# Unconditional training (optional comparison)
python train_dot_mesh.py --model_mode small --condition_mode unconditional --epochs 100

# Medium model
python train_dot_mesh.py --model_mode medium --condition_mode conditional --epochs 100

# Full model (best quality, needs 20GB+ GPU)
python train_dot_mesh.py --model_mode full --condition_mode conditional --epochs 200
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
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 16 (small) | Batch size per GPU |
| `--gpus` | 1 | Number of GPUs |
| `--n_train` | -1 | Training samples (-1=all) |
| `--data_dir` | `trash/data` | Path to train_codes.pt and val_codes.pt |
| `--out_base` | `trash/dot_runs` | Output directory base |
| `--plot_every` | 5/10 | Plot frequency: small=5, medium/full=10 |

## Conditional vs Unconditional

DoT supports two training modes:

**Conditional** (`--condition_mode conditional`):
- Prepends [BOS, class_token] to input sequences
- Model learns to generate based on class label
- Required for controlled generation (e.g., "generate a chair")
- **Use this for semantic channel experiments**

**Unconditional** (`--condition_mode unconditional`):
- Prepends only [BOS] token
- Model generates without class guidance
- Useful for comparing with conditional baseline
- Generates based on learned overall distribution

**Which to use?**
- For semantic channel evaluation: Use **conditional**
- For ablation studies: Train both and compare

## Custom Paths Example

```bash
python train_dot_mesh.py \
    --data_dir /path/to/data \
    --out_base /path/to/output \
    --model_mode small \
    --epochs 100 \
    --batch_size 16
```

## Outputs

```
trash/dot_runs/dot_<model_mode>_<condition_mode>_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── dot-epoch=XX-val_loss=X.XXXX.ckpt   ← Best checkpoints
│   └── dot_final.pt                          ← Final model
├── plots/
│   ├── curves_epXXXX.png                     ← Training curves
│   ├── gen_hist_epXXXX.png                   ← Per-class histograms (or sample histograms if unconditional)
│   └── code_dist_epXXXX.png                  ← Real vs generated distribution
└── report.json                               ← Training summary
```

## Model Configurations

| Mode | n_embd | n_head | n_layer | VRAM | Epochs | Batch |
|------|--------|--------|---------|------|--------|-------|
| **small** | 128 | 4 | 3 | ~4GB | 100 | 16 |
| **medium** | 256 | 8 | 4 | ~8GB | 100 | 12 |
| **full** | 512 | 8 | 6 | ~20GB | 200 | 8 |

## Architecture

- **Autoregressive**: Predicts next token given previous tokens
- **Class-conditional**: Prepends [BOS, class_token] to input
- **Block size**: 4096 tokens (full sequence)
- **Vocabulary**: 256 codes + 41 special tokens (BOS + 40 classes)

## Generation Modes

DoT supports two generation modes:
1. **Class ID only**: Generate from [BOS, class_token]
2. **Prefix completion**: Complete from partial sequence

Both use the same trained model.

## GPU Memory Guide

| Model | VRAM | Use When |
|-------|------|----------|
| small (128d) | ~4GB | Your GPU (~11GB), testing |
| medium (256d) | ~8GB | Balanced speed/quality |
| full (512d) | ~20GB | Best quality, production |
