# Training Scripts

## Overview

These scripts train individual models or all models together.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_classifier.py` | Train token classifier | `python train_classifier.py --tokens_path path/to/train_codes.pt --mode conditional` |
| `train_sedd.py` | Train SEDD diffusion model | `python train_sedd.py --mode small --epochs 200` |
| `train_dot.py` | Train DoT (NanoGPT) model | `python train_dot.py --mode small --epochs 100` |
| `train_all.py` | Train ALL models sequentially | `python train_all.py --data_dir trash/data --out_dir trash/training_runs` |

## Configuration

Edit `config.yaml` or pass CLI arguments to customize:
- Epochs, batch size, learning rate
- Model size (small/medium/full)
- GPU settings

## Outputs

All models save to:
- Checkpoints: `{out_dir}/{model}/checkpoints/`
- Logs: `{out_dir}/{model}/training.log`
- Plots: `{out_dir}/{model}/plots/`
- Reports: `{out_dir}/{model}/results.json`
