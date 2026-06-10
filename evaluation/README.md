# Evaluation Scripts

Scripts for testing generative models and semantic channel experiments.

## Quick Run (Unified)

```bash
cd semantic_channel
python run_all.py --epochs 1  # Quick test all models
```

## Individual Evaluation

### 1. Token Generation (SEDD)
```bash
cd ..
python generate_tokens.py \
    --ckpt ../trash/sedd_runs/<run>/checkpoints/best.ckpt \
    --mode conditional \
    --n_samples 1000
```

### 2. Semantic Channel Testing
```bash
cd semantic_channel
python mesh_transmit_semantic_channel.py \
    --classifier_ckpt <path> \
    --sedd_ckpt <path> \
    --dot_ckpt <path> \
    --real_codes_path ../../trash/data/val_codes.pt
```

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_tokens.py` | Generate tokens from SEDD |
| `generate_tokens_parallel.py` | Multi-GPU generation |
| `semantic_channel/mesh_transmit_semantic_channel.py` | Puncture + refill testing |
| `semantic_channel/run_all.py` | Full pipeline runner |

## Args

### generate_tokens.py
- `--ckpt` - SEDD checkpoint path
- `--mode` - `conditional` or `unconditional`
- `--n_samples` - Number to generate
- `--out_dir` - Output directory

### mesh_transmit_semantic_channel.py
- `--classifier_ckpt` - Classifier checkpoint
- `--sedd_ckpt` - SEDD checkpoint
- `--dot_ckpt` - DoT checkpoint
- `--real_codes_path` - Validation codes
- `--sedd_steps` - Diffusion steps (default: 50)
- `--skip_dot_full_gen` - Skip slow DoT full generation

## Output

- Results: `trash/semantic_channel_results/`
- Plots: Accuracy vs missing % curves
