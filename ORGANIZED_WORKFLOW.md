# Organized Workflow Guide

This guide explains the new organized directory structure and how to use it.

## 📁 Directory Structure

```
MeshGeneration/
├── scripts/                    # ⭐ All executable scripts here
│   ├── training/              # Training scripts only
│   │   ├── train_all.py     # Train all models
│   │   └── README.md
│   ├── inference/             # Inference & evaluation
│   │   ├── semantic_channel.py  # Main evaluation script
│   │   └── README.md
│   ├── testing/               # Quick tests
│   │   ├── quick_test.py      # 5-min sanity check
│   │   └── README.md
│   └── pipeline/              # End-to-end pipelines
│       ├── run_all.py         # Train + Evaluate
│       └── README.md
│
├── models/                    # Model implementations
│   ├── classifier/
│   ├── diffusion/            # SEDD
│   └── autoregressive/       # DoT
│
├── evaluation/                # Evaluation tools
│   └── semantic_channel/
│       └── mesh_transmit_semantic_channel.py  # (original implementation)
│
├── mesh_vqvae/               # VQ-VAE encoder/decoder
└── trash/                    # All outputs (created automatically)
    ├── data/                 # train_codes.pt, val_codes.pt
    ├── training_runs/        # Model checkpoints
    ├── semantic_channel_results/  # Evaluation outputs
    └── pipeline_results/     # Full pipeline outputs
```

---

## 🚀 Quick Usage

### 1. Test Everything Works (5 minutes)
```bash
cd scripts/testing
python quick_test.py
```

### 2. Full Training (hours)
```bash
cd scripts/training
python train_all.py --data_dir trash/data --out_dir trash/training_runs
```

### 3. Run Semantic Channel Evaluation (on trained models)
```bash
cd scripts/inference
python semantic_channel.py \
    --classifier_ckpt trash/training_runs/classifier/checkpoints/best.ckpt \
    --sedd_ckpt trash/training_runs/sedd \
    --dot_ckpt trash/training_runs/dot
```

### 4. Complete Pipeline (Train + Evaluate)
```bash
cd scripts/pipeline
python run_all.py
```

---

## 📊 Where Are My Results?

| Experiment Type | Output Location |
|-----------------|-----------------|
| **Training** | `trash/training_runs/{model}/` |
| **Checkpoints** | `trash/training_runs/{model}/checkpoints/` |
| **Training Plots** | `trash/training_runs/{model}/plots/` |
| **Semantic Channel** | `trash/semantic_channel_results/` |
| **Main Plot** | `trash/semantic_channel_results/plots/accuracy_vs_missing_tokens.png` |
| **Full Pipeline** | `trash/pipeline_results/` |

---

## 🔍 Finding Things

### I want to...

| Task | Script Location |
|------|-----------------|
| Train classifier only | `scripts/training/train_all.py --skip_sedd --skip_dot` |
| Train SEDD only | `scripts/training/train_all.py --skip_classifier --skip_dot` |
| Train DoT only | `scripts/training/train_all.py --skip_classifier --skip_sedd` |
| Generate SEDD samples | `scripts/inference/generate_sedd.py` |
| Generate DoT samples | `scripts/inference/generate_dot.py` |
| Plot semantic channel | `scripts/inference/semantic_channel.py` |
| Test quickly | `scripts/testing/quick_test.py` |

---

## 🆚 Old vs New Scripts

| Old (Root Folder) | New (Organized) | Purpose |
|-------------------|-----------------|---------|
| `train_all_models.py` | `scripts/training/train_all.py` | Train all models |
| `test_all_models.py` | `scripts/testing/quick_test.py` | Quick test |
| `test_inference.py` | `scripts/inference/semantic_channel.py` | Run evaluation |
| `generate_semantic_channel_plot.py` | `scripts/inference/semantic_channel.py` | (integrated) |
| `evaluation/semantic_channel/run_all.py` | `scripts/pipeline/run_all.py` | Full pipeline |

---

## 📝 Configuration

Edit training parameters in:
- `scripts/training/train_all.py` (TRAIN_CONFIG dict)
- Or pass CLI arguments: `--epochs`, `--batch_size`, etc.

---

## ⚡ TL;DR

```bash
# You only need to remember 3 scripts:

# 1. Test (5 min)
python scripts/testing/quick_test.py

# 2. Train (hours)
python scripts/training/train_all.py

# 3. Evaluate (if not using pipeline)
python scripts/inference/semantic_channel.py

# OR do it all at once:
python scripts/pipeline/run_all.py
```
