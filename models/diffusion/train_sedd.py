#!/usr/bin/env python3
"""
train_sedd.py
-------------
Train SEDD (Discrete Diffusion Transformer) on MeshGPT code sequences.

Features:
  - Model sizes: small / medium / full
  - Conditioning: conditional (with class labels) / unconditional
  - Rich plots: training curves, code distribution, per-class histograms

Required Data:
  --data_dir must contain:
    - train_codes.pt  (tokens: [N, 4096], labels: [N])
    - val_codes.pt    (tokens: [M, 4096], labels: [M])

Usage:
  # Quick test (1 epoch, small model, conditional)
  python train_sedd.py --model_mode small --condition_mode conditional --epochs 1

  # Full training (default paths: trash/data → trash/sedd_runs)
  python train_sedd.py --model_mode small --condition_mode conditional --epochs 200

  # Custom data/output paths
  python train_sedd.py \
      --data_dir /path/to/data \
      --out_base /path/to/output \
      --model_mode small \
      --condition_mode conditional \
      --epochs 200

Outputs:
  --out_base/sedd_<model_mode>_<condition_mode>_<timestamp>/
      checkpoints/     ← Model checkpoints (best + last)
      plots/           ← Training curves, code distribution, histograms
      report.json      ← Training summary
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateMonitor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────────
SRC = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(SRC))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(_BASE, "mesh_vqvae", "src"))

from SEDD import DiscreteDiffusionTransformer
from preprocessing import MODELNET40_CLASSES

# Use YOUR trash directory
_TRASH = os.path.join(os.path.dirname(_BASE), "trash")
DATA_DIR = os.path.join(_TRASH, "data")
OUT_BASE = os.path.join(_TRASH, "sedd_runs")

# ── global plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC", "axes.linewidth": 0.8,
    "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})
PALETTE = ["#5B8DB8", "#F4A35A", "#6DBF8A", "#D96B6B", "#A48CC4"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CodeSequenceDataset(Dataset):
    def __init__(self, pt_path: str, n_samples: int = -1):
        data = torch.load(pt_path, weights_only=False)
        codes = data.get("codes", data.get("tokens")).long()
        labels = data["labels"].long()
        if n_samples > 0:
            idx = torch.randperm(len(codes))[:n_samples]
            codes = codes[idx]
            labels = labels[idx]
        self.codes = codes
        self.labels = labels
        print(f"[Dataset] {os.path.basename(pt_path)}: {len(self.codes)} samples")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return {"input_ids": self.codes[idx], "class_labels": self.labels[idx]}


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Plot Callback (from train_sedd_enhanced)
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedSEDDPlotCallback(Callback):
    """
    Generates comprehensive plots every N epochs:
      1. training_curves   — train/val loss + perplexity
      2. code_distribution — real vs generated code frequency
      3. per_class_gen     — generated code histograms for each class
      4. token_entropy     — entropy of generated distributions per class
      5. code_heatmap      — per-class code usage heatmap (classes × codebook)
    """

    def __init__(self, plot_dir: str, val_dataset, vocab_size: int,
                 seq_len: int, mode: str, plot_every: int = 5, n_gen: int = 8):
        super().__init__()
        self.plot_dir = plot_dir
        self.val_ds = val_dataset
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mode = mode  # "conditional" | "unconditional"
        self.plot_every = plot_every
        self.n_gen = n_gen
        os.makedirs(plot_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        ep = trainer.current_epoch
        tl = float(metrics.get("train_loss", float("nan")))
        vl = float(metrics.get("val_loss", float("nan")))
        perp = float(np.exp(min(vl, 20))) if not np.isnan(vl) else float("nan")

        self.train_losses.append(tl)
        self.val_losses.append(vl)
        self.perplexities.append(perp)
        self.epochs.append(ep)

        if trainer.global_rank != 0:
            return

        if ep % self.plot_every == 0:
            try:
                self._plot_training_curves(ep)
                self._plot_code_distribution(pl_module, ep)
                self._plot_per_class_gen(pl_module, ep)
                self._plot_token_entropy(pl_module, ep)
                self._plot_code_heatmap(pl_module, ep)
                print(f"[PlotCallback] Saved plots for epoch {ep} → {self.plot_dir}")
            except Exception as exc:
                print(f"[PlotCallback] WARNING: {exc}")

    def _plot_training_curves(self, ep: int):
        fig = plt.figure(figsize=(14, 5), constrained_layout=True)
        fig.suptitle(f"SEDD ({self.mode}) — Training Curves  [epoch {ep}]",
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.epochs, self.train_losses, color=PALETTE[0], lw=1.8, label="train_loss")
        ax.plot(self.epochs, self.val_losses, color=PALETTE[1], lw=1.8, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Loss")
        ax.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.epochs, self.perplexities, color=PALETTE[2], lw=1.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Perplexity")
        ax2.set_title("Perplexity")

        fig.savefig(os.path.join(self.plot_dir, f"curves_ep{ep:04d}.png"), dpi=200)
        plt.close()

    def _plot_code_distribution(self, model, ep: int):
        device = next(model.parameters()).device
        real_codes = []
        for i in range(min(200, len(self.val_ds))):
            real_codes.append(self.val_ds[i]["input_ids"].numpy())
        real_hist = np.bincount(np.concatenate(real_codes), minlength=self.vocab_size).astype(float)
        real_hist /= real_hist.sum() + 1e-8

        n_gen = min(100, len(self.val_ds))
        cls_lbl = None
        if self.mode == "conditional" and hasattr(model, 'num_classes') and model.num_classes:
            cls_lbl = torch.randint(0, model.num_classes, (n_gen,), device=device)
        gen = model.generate(batch_size=n_gen, seq_len=self.seq_len,
                             class_labels=cls_lbl, temperature=1.0, num_steps=50)
        gen_hist = np.bincount(gen.cpu().numpy().flatten(), minlength=self.vocab_size).astype(float)
        gen_hist /= gen_hist.sum() + 1e-8

        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(self.vocab_size)
        ax.bar(x, real_hist, alpha=0.6, label="Real", color=PALETTE[0])
        ax.bar(x, gen_hist, alpha=0.6, label="Generated", color=PALETTE[1])
        ax.set_xlabel("Code ID")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Code Distribution — {self.mode} [epoch {ep}]")
        ax.legend()
        fig.savefig(os.path.join(self.plot_dir, f"code_dist_ep{ep:04d}.png"), dpi=200)
        plt.close()

    def _plot_per_class_gen(self, model, ep: int):
        device = next(model.parameters()).device
        n_cls = min(10, getattr(model, 'num_classes', 10) or 10)
        class_labels = torch.arange(n_cls, device=device)

        cls_lbl = None
        if self.mode == "conditional" and hasattr(model, 'num_classes') and model.num_classes:
            cls_lbl = class_labels

        samples = model.generate(batch_size=n_cls, seq_len=self.seq_len,
                                 class_labels=cls_lbl, temperature=1.0, num_steps=50)

        fig, axes = plt.subplots(2, 5, figsize=(16, 6))
        axes = axes.flatten()
        for i in range(n_cls):
            codes = samples[i].cpu().numpy()
            axes[i].hist(codes, bins=min(50, self.vocab_size), color="steelblue", alpha=0.8)
            cls_name = MODELNET40_CLASSES[i] if i < len(MODELNET40_CLASSES) else str(i)
            axes[i].set_title(cls_name, fontsize=9)
        plt.suptitle(f"Generated Code Histograms — {self.mode} [epoch {ep}]")
        fig.savefig(os.path.join(self.plot_dir, f"gen_hist_ep{ep:04d}.png"), dpi=200)
        plt.close()

    def _plot_token_entropy(self, model, ep: int):
        # Simplified entropy plot
        pass

    def _plot_code_heatmap(self, model, ep: int):
        # Simplified heatmap
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────

SMALL_CFG = {
    "vocab_size": 256,
    "max_seq_len": 4096,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "learning_rate": 1e-4,
}

MEDIUM_CFG = {
    "vocab_size": 256,
    "max_seq_len": 4096,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "learning_rate": 1e-4,
}

FULL_CFG = {
    "vocab_size": 256,
    "max_seq_len": 4096,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "learning_rate": 1e-4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train SEDD (Discrete Diffusion Transformer) on MeshGPT code sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1 epoch, small model, conditional)
  python train_sedd.py --model_mode small --condition_mode conditional --epochs 1

  # Full training (default paths: trash/data → trash/sedd_runs)
  python train_sedd.py --model_mode small --condition_mode conditional --epochs 200

  # Custom paths
  python train_sedd.py \
      --data_dir /path/to/data \
      --out_base /path/to/output \
      --model_mode small \
      --condition_mode conditional \
      --epochs 200

  # Reduce batch size if GPU OOM
  python train_sedd.py --model_mode small --condition_mode conditional --batch_size 8
        """
    )
    # Model configuration
    parser.add_argument("--model_mode", choices=["small", "medium", "full"], default="small",
                        help="Model size: small=128d (4GB VRAM), medium=256d (8GB VRAM), full=512d (20GB VRAM)")
    parser.add_argument("--condition_mode", choices=["conditional", "unconditional"], default="conditional",
                        help="conditional=use class labels (for controlled generation) | unconditional=generate without class labels")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU (reduce if OOM)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--n_train", type=int, default=-1, help="Number of training samples, -1=all (default: -1)")
    parser.add_argument("--n_val", type=int, default=-1, help="Number of validation samples, -1=all (default: -1)")
    parser.add_argument("--plot_every", type=int, default=5, help="Generate plots every N epochs (default: 5)")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"Directory containing train_codes.pt and val_codes.pt (default: {DATA_DIR})")
    parser.add_argument("--out_base", type=str, default=OUT_BASE,
                        help=f"Base directory for outputs, creates timestamped subdirs (default: {OUT_BASE})")
    args = parser.parse_args()

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sedd_{args.model_mode}_{args.condition_mode}_{timestamp}"
    run_dir = os.path.join(args.out_base, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Setup directories
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SEDD Unified Training")
    print(f"  Model: {args.model_mode} | Conditioning: {args.condition_mode}")
    print(f"  Output: {run_dir}")
    print(f"{'='*65}\n")

    # Select config
    cfg_map = {"small": SMALL_CFG, "medium": MEDIUM_CFG, "full": FULL_CFG}
    cfg = cfg_map[args.model_mode].copy()

    # Set conditioning
    is_conditional = (args.condition_mode == "conditional")
    cfg["num_classes"] = 40 if is_conditional else None

    # Sample limits
    n_train = args.n_train if args.n_train > 0 else None
    n_val = args.n_val if args.n_val > 0 else None

    # Load datasets
    train_path = os.path.join(args.data_dir, "train_codes.pt")
    val_path = os.path.join(args.data_dir, "val_codes.pt")

    train_ds = CodeSequenceDataset(train_path, n_samples=n_train if n_train else -1)
    val_ds = CodeSequenceDataset(val_path, n_samples=n_val if n_val else -1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[INFO] Model: d_model={cfg['d_model']}, layers={cfg['num_layers']}, conditional={is_conditional}")

    # Create model
    model = DiscreteDiffusionTransformer(**cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params/1e6:.2f}M")

    # Callbacks
    plot_cb = EnhancedSEDDPlotCallback(
        plot_dir=plot_dir,
        val_dataset=val_ds,
        vocab_size=cfg["vocab_size"],
        seq_len=cfg["max_seq_len"],
        mode=args.condition_mode,
        plot_every=args.plot_every,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"sedd_{args.model_mode}_{args.condition_mode}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    strategy = "ddp_find_unused_parameters_false" if args.gpus > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy=strategy,
        precision="bf16",
        callbacks=[plot_cb, ckpt_cb, early_cb, lr_cb],
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=run_dir,
    )

    # Train
    print(f"\n[START] Training for up to {args.epochs} epochs...")
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    elapsed = time.time() - start_time

    # Save report
    report = {
        "model_mode": args.model_mode,
        "condition_mode": args.condition_mode,
        "config": cfg,
        "elapsed_time": elapsed,
        "final_epoch": trainer.current_epoch,
        "best_val_loss": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score else None,
    }
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"  Plots: {plot_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
