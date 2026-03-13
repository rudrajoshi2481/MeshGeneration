"""
train_sedd.py
--------------
Trains SEDD (DiscreteDiffusionTransformer) on MeshGPT code sequences.

Two modes:
  --mode small   : tiny model, 200 samples, quick hyperparameter validation
  --mode full    : full model, all samples, 8 GPUs

Outputs → /data/joshi/MESHGPT/new_implementation/trash/sedd_runs/<run_id>/
  checkpoints/
  logs/train.log
  plots/
  report.json

Usage:
  # Step 1: extract codes (run once)
  python extract_codes.py

  # Step 2a: small test
  python train_sedd.py --mode small

  # Step 2b: full run (after small test passes)
  python train_sedd.py --mode full
"""

import os
import sys
import json
import argparse
import time
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

SEDD_PATH = "/data/joshi/yagnas_stuff/MLopsThesis/Models/SEDD.py"
sys.path.insert(0, os.path.dirname(SEDD_PATH))
from SEDD import DiscreteDiffusionTransformer, DiscreteNoiseSchedule

from preprocessing import MODELNET40_CLASSES

DATA_DIR = "/data/joshi/MESHGPT/new_implementation/trash/sedd_data"
OUT_BASE  = "/data/joshi/MESHGPT/new_implementation/trash/sedd_runs"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset: wraps pre-extracted code sequences
# ─────────────────────────────────────────────────────────────────────────────

class CodeSequenceDataset(Dataset):
    """Simple dataset over pre-extracted MeshGPT code sequences."""

    def __init__(self, pt_path: str, n_samples: int = -1):
        data = torch.load(pt_path, weights_only=False)
        codes  = data["codes"].long()    # [N, 4096]
        labels = data["labels"].long()   # [N]
        if n_samples > 0:
            idx = torch.randperm(len(codes))[:n_samples]
            codes  = codes[idx]
            labels = labels[idx]
        self.codes  = codes
        self.labels = labels
        print(f"[Dataset] {pt_path.split('/')[-1]}: {len(self.codes)} samples")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return {
            "input_ids":    self.codes[idx],
            "class_labels": self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting callbacks
# ─────────────────────────────────────────────────────────────────────────────

class SEDDPlotCallback(Callback):
    """Logs training curves and generation samples every N epochs."""

    def __init__(self, plot_dir: str, val_dataset, vocab_size: int,
                 seq_len: int, plot_every: int = 5, n_gen: int = 8):
        self.plot_dir   = plot_dir
        self.val_ds     = val_dataset
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.plot_every = plot_every
        self.n_gen      = n_gen
        os.makedirs(plot_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses   = []
        self.epochs       = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        ep = trainer.current_epoch
        tl = float(metrics.get("train_loss", float("nan")))
        vl = float(metrics.get("val_loss",   float("nan")))
        self.train_losses.append(tl)
        self.val_losses.append(vl)
        self.epochs.append(ep)

        if ep % self.plot_every == 0:
            self._plot_curves(ep)
            self._plot_generation(pl_module, ep)
            self._plot_code_distribution(pl_module, ep)

    def _plot_curves(self, ep: int):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(self.epochs, self.train_losses, label="train_loss")
        ax.plot(self.epochs, self.val_losses,   label="val_loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("SEDD Training Curves"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"curves_ep{ep:04d}.png"), dpi=120)
        plt.close()

    @torch.no_grad()
    def _plot_generation(self, model, ep: int):
        """Generate one sample per class (first 10 classes) and plot code histograms."""
        device = next(model.parameters()).device
        n_cls  = min(10, model.num_classes or 10)
        class_labels = torch.arange(n_cls, device=device)

        samples = model.generate(
            batch_size=n_cls,
            seq_len=self.seq_len,
            class_labels=class_labels,
            temperature=1.0,
            num_steps=50,
        )  # [n_cls, seq_len]

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        axes = axes.flatten()
        for i in range(n_cls):
            codes = samples[i].cpu().numpy()
            axes[i].hist(codes, bins=min(50, self.vocab_size),
                         color="steelblue", alpha=0.8, edgecolor="none")
            cls_name = MODELNET40_CLASSES[i] if i < len(MODELNET40_CLASSES) else str(i)
            axes[i].set_title(cls_name, fontsize=9)
            axes[i].set_xlabel("Code ID"); axes[i].set_ylabel("Count")
        plt.suptitle(f"Generated Code Histograms (epoch {ep})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"gen_hist_ep{ep:04d}.png"), dpi=120)
        plt.close()

    @torch.no_grad()
    def _plot_code_distribution(self, model, ep: int):
        """Compare real vs generated code usage distribution."""
        device = next(model.parameters()).device

        # Real codes from val set (up to 200 samples)
        real_codes = []
        for i in range(min(200, len(self.val_ds))):
            real_codes.append(self.val_ds[i]["input_ids"])
        real_codes = torch.stack(real_codes).numpy()
        real_hist  = np.bincount(real_codes.flatten(), minlength=self.vocab_size).astype(float)
        real_hist /= real_hist.sum()

        # Generated codes (200 samples, random classes)
        n_gen = min(200, len(self.val_ds))
        labels = torch.randint(0, model.num_classes or 40, (n_gen,), device=device)
        gen = model.generate(batch_size=n_gen, seq_len=self.seq_len,
                             class_labels=labels, temperature=1.0, num_steps=50)
        gen_hist = np.bincount(gen.cpu().numpy().flatten(),
                               minlength=self.vocab_size).astype(float)
        gen_hist /= gen_hist.sum()

        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(self.vocab_size)
        ax.bar(x, real_hist, alpha=0.6, label="Real", color="blue",  width=1.0)
        ax.bar(x, gen_hist,  alpha=0.6, label="Generated", color="orange", width=1.0)
        ax.set_xlabel("Code ID"); ax.set_ylabel("Frequency")
        ax.set_title(f"Real vs Generated Code Distribution (epoch {ep})")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"code_dist_ep{ep:04d}.png"), dpi=120)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Config presets
# ─────────────────────────────────────────────────────────────────────────────

SMALL_CFG = dict(
    vocab_size     = 256,
    max_seq_len    = 4096,
    d_model        = 128,    # tiny
    nhead          = 4,
    num_layers     = 3,
    dim_feedforward= 512,
    dropout        = 0.1,
    num_classes    = 40,
    mask_id        = 256,
    num_timesteps  = 1000,
    schedule_type  = "cosine",
    learning_rate  = 5e-4,
    beta1          = 0.9,
    beta2          = 0.99,
    weight_decay   = 0.01,
)

MEDIUM_CFG = dict(
    vocab_size     = 256,
    max_seq_len    = 4096,
    d_model        = 256,
    nhead          = 8,
    num_layers     = 4,
    dim_feedforward= 1024,
    dropout        = 0.1,
    num_classes    = 40,
    mask_id        = 256,
    num_timesteps  = 1000,
    schedule_type  = "cosine",
    learning_rate  = 3e-4,
    beta1          = 0.9,
    beta2          = 0.99,
    weight_decay   = 0.01,
)

FULL_CFG = dict(
    vocab_size     = 256,
    max_seq_len    = 4096,
    d_model        = 512,
    nhead          = 8,
    num_layers     = 6,
    dim_feedforward= 2048,
    dropout        = 0.1,
    num_classes    = 40,
    mask_id        = 256,
    num_timesteps  = 1000,
    schedule_type  = "cosine",
    learning_rate  = 1e-4,
    beta1          = 0.9,
    beta2          = 0.99,
    weight_decay   = 0.01,
)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_generation(model, val_ds, vocab_size: int, seq_len: int,
                         n_samples: int = 200, device="cuda"):
    """
    Compute per-class code recall:
      for each class, generate N samples, compute histogram overlap with real samples.
    Returns avg_overlap (0-1, higher=better).
    """
    model.eval()
    n_classes = model.num_classes or 40
    n_per_cls = max(1, n_samples // n_classes)

    # Real code histograms per class
    real_hists = {}
    for i in range(len(val_ds)):
        item = val_ds[i]
        c = int(item["class_labels"])
        if c not in real_hists:
            real_hists[c] = np.zeros(vocab_size)
        real_hists[c] += np.bincount(item["input_ids"].numpy(), minlength=vocab_size)

    overlaps = []
    for c in range(n_classes):
        if c not in real_hists:
            continue
        real_h = real_hists[c] / (real_hists[c].sum() + 1e-8)
        labels = torch.full((n_per_cls,), c, dtype=torch.long, device=device)
        gen = model.generate(batch_size=n_per_cls, seq_len=seq_len,
                             class_labels=labels, temperature=1.0, num_steps=50)
        gen_h = np.bincount(gen.cpu().numpy().flatten(),
                            minlength=vocab_size).astype(float)
        gen_h /= gen_h.sum() + 1e-8
        overlap = np.minimum(real_h, gen_h).sum()
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def save_report(run_dir: str, mode: str, cfg: dict, results: dict):
    report = {"mode": mode, "config": cfg, "results": results,
              "timestamp": datetime.now().isoformat()}
    path = os.path.join(run_dir, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(mode: str, run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, "plots")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "train.log")

    print(f"\n{'='*60}")
    print(f"  SEDD Training — mode={mode}")
    print(f"  run_dir = {run_dir}")
    print(f"{'='*60}\n")

    # ── mode settings ─────────────────────────────────────────────────────
    if mode == "small":
        cfg         = SMALL_CFG
        n_train     = 200
        n_val       = 50
        max_epochs  = 30
        batch_size  = 16
        num_gpus    = 1
        plot_every  = 5
        num_workers = 4
    elif mode == "medium":
        cfg         = MEDIUM_CFG
        n_train     = 800
        n_val       = 200
        max_epochs  = 50
        batch_size  = 32
        num_gpus    = 1
        plot_every  = 5
        num_workers = 8
    else:  # full
        cfg         = FULL_CFG
        n_train     = -1   # all
        n_val       = -1
        max_epochs  = 200
        batch_size  = 64   # per GPU
        num_gpus    = 8
        plot_every  = 10
        num_workers = 8

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = CodeSequenceDataset(os.path.join(DATA_DIR, "train_codes.pt"), n_train)
    val_ds   = CodeSequenceDataset(os.path.join(DATA_DIR, "val_codes.pt"),   n_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    n_params_approx = (cfg["d_model"] ** 2 * cfg["num_layers"] * 4) / 1e6
    print(f"[INFO] Model config: d_model={cfg['d_model']}, layers={cfg['num_layers']}, "
          f"~{n_params_approx:.1f}M params (rough estimate)")
    print(f"[INFO] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"[INFO] GPUs: {num_gpus}, batch_size/GPU: {batch_size}, epochs: {max_epochs}")

    # ── model ─────────────────────────────────────────────────────────────
    model = DiscreteDiffusionTransformer(**cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Actual model parameters: {total_params/1e6:.2f}M")

    # ── callbacks ─────────────────────────────────────────────────────────
    device_for_plot = "cuda" if torch.cuda.is_available() else "cpu"
    plot_cb = SEDDPlotCallback(
        plot_dir=plot_dir,
        val_dataset=val_ds,
        vocab_size=cfg["vocab_size"],
        seq_len=cfg["max_seq_len"],
        plot_every=plot_every,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="sedd-{epoch:04d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=20 if mode == "full" else 10,
        mode="min",
        verbose=True,
    )

    # ── trainer ───────────────────────────────────────────────────────────
    strategy = "ddp_find_unused_parameters_false" if num_gpus > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        strategy=strategy,
        precision="bf16",
        callbacks=[plot_cb, ckpt_cb, early_stop_cb],
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=run_dir,
        gradient_clip_val=1.0,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    elapsed = time.time() - t0

    # ── post-training: rank 0 only ─────────────────────────────────────────
    if trainer.global_rank != 0:
        return {}

    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min")

    best_ckpt = ckpt_cb.best_model_path
    print(f"[INFO] Best checkpoint: {best_ckpt}")

    eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    avg_overlap = 0.0
    if best_ckpt and os.path.exists(best_ckpt):
        print("[INFO] Running generation quality evaluation ...")
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        best_model = DiscreteDiffusionTransformer.load_from_checkpoint(
            best_ckpt, map_location=eval_device
        )
        torch.load = _orig
        best_model.eval().to(eval_device)

        avg_overlap = evaluate_generation(
            best_model, val_ds, cfg["vocab_size"], cfg["max_seq_len"],
            n_samples=min(200, len(val_ds) * 5), device=eval_device
        )
        print(f"[INFO] Avg code overlap (real vs generated): {avg_overlap:.4f}")

        plot_cb._plot_generation(best_model, ep=trainer.current_epoch)
        plot_cb._plot_code_distribution(best_model, ep=trainer.current_epoch)
    else:
        print("[WARN] No best checkpoint found, skipping eval plots.")

    # ── save report ────────────────────────────────────────────────────────
    results = {
        "best_val_loss":    float(trainer.callback_metrics.get("val_loss", -1)),
        "total_params_M":   round(total_params / 1e6, 3),
        "train_epochs":     trainer.current_epoch,
        "train_time_min":   round(elapsed / 60, 1),
        "avg_code_overlap": round(avg_overlap, 4),
        "best_ckpt":        best_ckpt,
        "n_train":          len(train_ds),
        "n_val":            len(val_ds),
    }
    save_report(run_dir, mode, cfg, results)

    print(f"\n{'='*60}")
    print(f"  DONE — mode={mode}")
    print(f"  best_val_loss  : {results['best_val_loss']:.4f}")
    print(f"  avg_overlap    : {results['avg_code_overlap']:.4f}")
    print(f"  params         : {results['total_params_M']}M")
    print(f"  output         : {run_dir}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["small", "medium", "full"], default="small")
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_BASE, f"sedd_{args.mode}_{ts}")

    results = run(args.mode, run_dir)
    return results


if __name__ == "__main__":
    main()
