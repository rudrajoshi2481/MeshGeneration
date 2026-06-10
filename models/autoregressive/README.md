# DoT (Decoder-only Transformer)

Autoregressive model for token sequence generation (NanoGPT architecture).

## Quick Run

```bash
# Train DoT (small config, 1 GPU)
python train_dot_mesh.py --mode small --epochs 100 --gpus 1 --batch_size 16

# Quick test (1 epoch)
python train_dot_mesh.py --mode small --epochs 1 --gpus 1 --batch_size 16
```

## Config Changes

Edit `train_dot_mesh.py` lines 180-200:

```python
# SMALL config
n_embd=128,       # Embedding dimension
n_head=4,         # Attention heads
n_layer=3,        # Transformer layers
block_size=352,   # Context window (seq length for training)
dropout=0.1
additional_vocab=41  # BOS + 40 classes
```

## Args

- `--mode` - `small` (128d) / `full` (256d)
- `--epochs` - Training epochs (default: 100)
- `--gpus` - Number of GPUs (default: 1)
- `--batch_size` - Batch size (default: 16)
- `--n_train` - Samples to use (-1 for all)

## Output

- Checkpoints: `trash/dot_runs/<run_id>/checkpoints/`
- Best val_loss checkpoint saved automatically

## Generation Modes

DoT supports two generation modes:
1. **Class ID only**: Generate from [BOS, class_token]
2. **Prefix completion**: Complete from partial sequence

Both use the same trained model.
