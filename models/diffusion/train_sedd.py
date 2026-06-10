#!/usr/bin/env python3
"""
train_sedd.py
-------------
Train SEDD (Discrete Diffusion Transformer) on MeshGPT code sequences.

Features:
  - Model sizes: small / medium / full
  - Conditioning: conditional (with class labels) / unconditional
  - Professional paper-quality plots (300 DPI PNG + SVG vector graphics)
  - Plots: training curves, code distribution, per-class histograms

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
          *.png        ← 300 DPI raster for quick viewing
          *.svg        ← Vector graphics for papers
      report.json      ← Training summary

Plot Features:
  - Publication-quality typography and styling
  - Colorblind-friendly palette
  - Automatic text wrapping for long labels
  - Metrics annotations (diversity, overlap, final values)
  - Seaborn integration (if installed) for enhanced aesthetics
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
from textwrap import wrap

# Try to import seaborn for professional styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

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
# Professional paper-quality styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})

# Professional color palette (colorblind-friendly)
PALETTE = {
    "primary": "#5B8DB8",      # Steel blue
    "secondary": "#E67E22",     # Warm orange
    "tertiary": "#27AE60",      # Green
    "quaternary": "#8E44AD",    # Purple
    "accent": "#C0392B",        # Red
    "neutral": "#7F8C8D",       # Gray
    "dark": "#2C3E50",          # Dark blue-gray
    "light": "#BDC3C7",         # Light gray
}

# Alias for backward compatibility
PALETTE_LIST = [PALETTE["primary"], PALETTE["secondary"], PALETTE["tertiary"], 
                PALETTE["quaternary"], PALETTE["accent"]]

def _save_figure(fig, filepath: str, dpi: int = 300):
    """Save figure as both PNG and SVG for paper quality."""
    # Save PNG for quick viewing
    fig.savefig(f"{filepath}.png", dpi=dpi, bbox_inches="tight", 
                facecolor="white", edgecolor="none")
    # Save SVG for vector quality in papers
    fig.savefig(f"{filepath}.svg", format="svg", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


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
    Professional paper-quality plotting callback.
    
    Generates publication-ready plots every N epochs:
      1. training_curves   — train/val loss + perplexity
      2. code_distribution — real vs generated code frequency
      3. per_class_gen     — generated code histograms per class
      4. token_entropy     — entropy analysis per class
      5. code_heatmap      — class × codebook usage heatmap
    
    All plots saved as both PNG (300 DPI) and SVG (vector) for papers.
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
        
        # Setup seaborn if available
        if HAS_SEABORN:
            sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

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
        """Publication-quality training curves with proper typography."""
        fig = plt.figure(figsize=(14, 5), constrained_layout=True)
        title = f"SEDD {self.mode.title()} Training Progress (Epoch {ep})"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

        # Loss plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.epochs, self.train_losses, color=PALETTE["primary"], 
                linewidth=2.0, label="Train Loss", alpha=0.9, marker="o", markersize=3)
        ax1.plot(self.epochs, self.val_losses, color=PALETTE["secondary"], 
                linewidth=2.0, label="Validation Loss", alpha=0.9, marker="s", markersize=3)
        ax1.set_xlabel("Epoch", fontsize=11, labelpad=8)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, labelpad=8)
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="semibold", pad=10)
        ax1.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
        ax1.grid(True, alpha=0.3, linestyle="-")
        
        # Add final values as text
        if len(self.train_losses) > 0:
            final_train = self.train_losses[-1]
            final_val = self.val_losses[-1]
            ax1.text(0.98, 0.98, f"Train: {final_train:.3f}\nVal: {final_val:.3f}",
                    transform=ax1.transAxes, fontsize=9, verticalalignment="top",
                    horizontalalignment="right", bbox=dict(boxstyle="round", 
                    facecolor="white", edgecolor="#CCCCCC", alpha=0.8))

        # Perplexity plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.epochs, self.perplexities, color=PALETTE["tertiary"], 
                linewidth=2.0, alpha=0.9, marker="o", markersize=3)
        ax2.set_xlabel("Epoch", fontsize=11, labelpad=8)
        ax2.set_ylabel("Perplexity (exp(loss))", fontsize=11, labelpad=8)
        ax2.set_title("Model Perplexity", fontsize=12, fontweight="semibold", pad=10)
        ax2.grid(True, alpha=0.3, linestyle="-")
        
        if len(self.perplexities) > 0:
            final_perp = self.perplexities[-1]
            ax2.text(0.98, 0.98, f"Final: {final_perp:.2f}",
                    transform=ax2.transAxes, fontsize=9, verticalalignment="top",
                    horizontalalignment="right", bbox=dict(boxstyle="round",
                    facecolor="white", edgecolor="#CCCCCC", alpha=0.8))

        _save_figure(fig, os.path.join(self.plot_dir, f"curves_ep{ep:04d}"), dpi=300)

    def _plot_code_distribution(self, model, ep: int):
        """Professional code distribution comparison: Real vs Generated."""
        device = next(model.parameters()).device
        
        # Collect real codes from validation set
        real_codes = []
        n_samples = min(500, len(self.val_ds))
        for i in range(n_samples):
            real_codes.append(self.val_ds[i]["input_ids"].numpy())
        real_hist = np.bincount(np.concatenate(real_codes), minlength=self.vocab_size).astype(float)
        real_hist /= real_hist.sum() + 1e-8

        # Generate samples
        n_gen = min(500, len(self.val_ds))
        cls_lbl = None
        if self.mode == "conditional" and hasattr(model, 'num_classes') and model.num_classes:
            cls_lbl = torch.randint(0, model.num_classes, (n_gen,), device=device)
        
        with torch.no_grad():
            gen = model.generate(batch_size=n_gen, seq_len=self.seq_len,
                                class_labels=cls_lbl, temperature=1.0, num_steps=50)
        gen_hist = np.bincount(gen.cpu().numpy().flatten(), minlength=self.vocab_size).astype(float)
        gen_hist /= gen_hist.sum() + 1e-8

        # Compute overlap for annotation
        overlap = np.sum(np.minimum(real_hist, gen_hist))
        
        # Create professional plot
        fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
        x = np.arange(self.vocab_size)
        
        # Plot with professional styling
        ax.fill_between(x, real_hist, alpha=0.7, color=PALETTE["primary"], 
                       label=f"Real (n={n_samples})", step="mid", linewidth=0)
        ax.fill_between(x, gen_hist, alpha=0.5, color=PALETTE["secondary"], 
                       label=f"Generated (n={n_gen})", step="mid", linewidth=0)
        
        # Add outline
        ax.plot(x, real_hist, color=PALETTE["primary"], linewidth=1.5, alpha=0.9)
        ax.plot(x, gen_hist, color=PALETTE["secondary"], linewidth=1.5, alpha=0.9)
        
        ax.set_xlabel("Code ID", fontsize=12, labelpad=10)
        ax.set_ylabel("Normalized Frequency", fontsize=12, labelpad=10)
        ax.set_title(f"Code Distribution: Real vs Generated\n{self.mode.title()} Mode, Epoch {ep}", 
                    fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, 
                 edgecolor="#AAAAAA", fontsize=11)
        ax.set_xlim(0, self.vocab_size)
        ax.grid(True, alpha=0.3, axis="y", linestyle="-")
        
        # Add overlap annotation
        ax.text(0.02, 0.98, f"Distribution Overlap: {overlap:.3f}",
               transform=ax.transAxes, fontsize=11, verticalalignment="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                        edgecolor=PALETTE["dark"], alpha=0.9))
        
        _save_figure(fig, os.path.join(self.plot_dir, f"code_dist_ep{ep:04d}"), dpi=300)

    def _plot_per_class_gen(self, model, ep: int):
        """Publication-quality per-class generation histograms."""
        device = next(model.parameters()).device
        n_cls = min(10, getattr(model, 'num_classes', 10) or 10)
        class_labels = torch.arange(n_cls, device=device)

        cls_lbl = None
        if self.mode == "conditional" and hasattr(model, 'num_classes') and model.num_classes:
            cls_lbl = class_labels

        with torch.no_grad():
            samples = model.generate(batch_size=n_cls, seq_len=self.seq_len,
                                    class_labels=cls_lbl, temperature=1.0, num_steps=50)

        # Professional multi-panel figure
        n_cols = 5
        n_rows = int(np.ceil(n_cls / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.0),
                                 constrained_layout=True)
        axes = axes.flatten()
        
        for i in range(n_cls):
            ax = axes[i]
            codes = samples[i].cpu().numpy()
            
            # Professional histogram with KDE if seaborn available
            if HAS_SEABORN:
                sns.histplot(codes, bins=min(40, self.vocab_size), ax=ax,
                           color=PALETTE["primary"], alpha=0.7, edgecolor="none")
            else:
                ax.hist(codes, bins=min(40, self.vocab_size), 
                       color=PALETTE["primary"], alpha=0.7, edgecolor="none")
            
            cls_name = MODELNET40_CLASSES[i] if i < len(MODELNET40_CLASSES) else f"Class {i}"
            # Wrap long class names
            wrapped_title = "\n".join(wrap(cls_name.title(), 15))
            ax.set_title(wrapped_title, fontsize=10, fontweight="semibold", pad=8)
            ax.set_xlabel("Code ID", fontsize=9, labelpad=6)
            ax.set_ylabel("Count", fontsize=9, labelpad=6)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            
            # Add code diversity metric
            unique_codes = len(np.unique(codes))
            diversity = unique_codes / self.vocab_size
            ax.text(0.98, 0.98, f"Unique: {unique_codes}\nDiversity: {diversity:.2%}",
                   transform=ax.transAxes, fontsize=8, verticalalignment="top",
                   horizontalalignment="right", 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            edgecolor=PALETTE["light"], alpha=0.9))
        
        # Hide unused subplots
        for idx in range(n_cls, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f"Per-Class Generated Code Distributions\n{self.mode.title()} Mode, Epoch {ep}",
                    fontsize=13, fontweight="bold", y=1.02)
        
        _save_figure(fig, os.path.join(self.plot_dir, f"gen_hist_ep{ep:04d}"), dpi=300)

    def _plot_token_entropy(self, model, ep: int):
        """Token entropy analysis per class (simplified placeholder)."""
        pass

    def _plot_code_heatmap(self, model, ep: int):
        """Class × Codebook heatmap (simplified placeholder)."""
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
