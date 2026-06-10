# Token Classifier

Trains a classifier on VQVAE token sequences to evaluate quality.

## Quick Run

```bash
python train_classifier.py \
    --tokens_path ../../trash/data/train_codes.pt \
    --mode conditional \
    --out_dir ../../trash/classifier_run \
    --epochs 50 \
    --gpus 1
```

## Test (1 Epoch)

```bash
python train_classifier.py \
    --tokens_path ../../trash/data/train_codes.pt \
    --mode conditional \
    --out_dir ../../trash/classifier_test \
    --epochs 1 \
    --gpus 1 \
    --batch_size 64
```

## Config Changes

Edit `train_classifier.py` lines 40-45:

```python
model = TokenClassifier(
    vocab_size=256,      # Codebook size
    seq_len=4096,        # Tokens per mesh
    embed_dim=256,       # Embedding dimension
    hidden_dim=512,      # MLP hidden size
    num_classes=40,      # ModelNet40 classes
    dropout=0.2          # Regularization
)
```

## Args

- `--tokens_path` - Path to train_codes.pt
- `--mode` - `conditional` or `unconditional`
- `--out_dir` - **Required.** Output directory for checkpoints
- `--epochs` - Training epochs (default: 50)
- `--gpus` - Number of GPUs (default: 1)
- `--batch_size` - Batch size (default: 64)
- `--lr` - Learning rate (default: 1e-3)

## Output

- Checkpoints: `trash/classifier_runs/<run_id>/checkpoints/`
- Plots: `trash/classifier_runs/<run_id>/plots/`
- Best acc: ~57% on real VQVAE tokens
