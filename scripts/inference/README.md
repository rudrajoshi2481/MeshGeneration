# Inference Scripts

## Overview

These scripts run inference on trained models and generate evaluation plots.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `evaluate_classifier.py` | Evaluate classifier on test data | `python evaluate_classifier.py --ckpt path/to/classifier.ckpt --tokens_path val_codes.pt` |
| `generate_sedd.py` | Generate samples with SEDD | `python generate_sedd.py --ckpt path/to/sedd.ckpt --n_samples 100` |
| `generate_dot.py` | Generate samples with DoT | `python generate_dot.py --ckpt path/to/dot.ckpt --n_samples 100` |
| `semantic_channel.py` | Full semantic channel evaluation | `python semantic_channel.py --classifier_ckpt ... --sedd_ckpt ... --dot_ckpt ...` |
| `run_all.py` | Complete pipeline: train + infer + evaluate | `python run_all.py` or `python run_all.py --eval_only` |

## Workflow

### 1. Quick Test (Already Trained Models)
```bash
# If you already have trained models, just run semantic channel evaluation
python scripts/inference/semantic_channel.py \
    --classifier_ckpt trash/training_runs/classifier/checkpoints/best.ckpt \
    --sedd_ckpt trash/training_runs/sedd/checkpoints/best.ckpt \
    --dot_ckpt trash/training_runs/dot/checkpoints/dot_final.ckpt \
    --real_codes_path trash/data/val_codes.pt \
    --out_dir trash/semantic_channel_results
```

### 2. Full Pipeline (Train → Infer → Evaluate)
```bash
# Train all models + run semantic channel evaluation
python scripts/pipeline/run_all.py

# Or skip training, just run inference with existing models
python scripts/pipeline/run_all.py --eval_only
```

## Outputs

All inference scripts save:
- Generated samples (`.pt` files with tokens/labels)
- Evaluation metrics (accuracy, per-class metrics)
- Plots (accuracy vs missing tokens, confusion matrices, etc.)
- JSON reports with detailed results

## Plot Locations

| Experiment | Plot Location |
|------------|---------------|
| Classifier eval | `{out_dir}/plots/` |
| SEDD generation | `{out_dir}/plots/` |
| DoT generation | `{out_dir}/plots/` |
| Semantic channel | `{out_dir}/plots/accuracy_vs_missing_tokens.png` |
