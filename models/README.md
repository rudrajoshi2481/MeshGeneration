# Models Directory

All model training scripts organized by architecture type.

## Quick Start

```bash
# Train all models with one command
cd evaluation/semantic_channel
python run_all.py --epochs 1  # Quick test (1 epoch each)
```

## Individual Training

### 1. Token Classifier
```bash
cd classifier
python train_classifier.py \
    --tokens_path ../../trash/data/train_codes.pt \
    --mode conditional \
    --epochs 50 \
    --gpus 1 \
    --batch_size 64
```

**Config**: Edit `train_classifier.py` lines 40-45:
- `vocab_size=256` - Codebook size
- `embed_dim=256` - Embedding dimension  
- `hidden_dim=512` - MLP hidden size
- `dropout=0.2` - Dropout rate

### 2. SEDD (Diffusion Model)
```bash
cd diffusion
python train_sedd.py \
    --mode small \
    --epochs 200 \
    --gpus 1 \
    --batch_size 16
```

**Config**: Edit `train_sedd.py` lines 25-50:
- `d_model=128` - Model dimension (small/medium/full presets)
- `nhead=4` - Attention heads
- `num_layers=3` - Transformer layers

### 3. DoT (Autoregressive)
```bash
cd autoregressive
python train_dot_mesh.py \
    --mode small \
    --epochs 100 \
    --gpus 1 \
    --batch_size 16
```

**Config**: Edit `train_dot_mesh.py` lines 180-200:
- `n_embd=128` - Embedding size
- `n_head=4` - Attention heads
- `n_layer=3` - Transformer layers
- `block_size=352` - Context window

## GPU Options

All scripts support:
- `--gpus 1` - Single GPU (default)
- `--gpus 2` - Multi-GPU (DDP)
- `--gpus -1` - All available GPUs

## Output Locations

- Classifier: `trash/classifier_runs/`
- SEDD: `trash/sedd_runs/`
- DoT: `trash/dot_runs/`

## Testing (1 Epoch)

```bash
# Quick sanity check for all models
python classifier/train_classifier.py --tokens_path trash/data/train_codes.pt --epochs 1 --gpus 1
python diffusion/train_sedd.py --mode small --epochs 1 --gpus 1
python autoregressive/train_dot_mesh.py --mode small --epochs 1 --gpus 1
```
