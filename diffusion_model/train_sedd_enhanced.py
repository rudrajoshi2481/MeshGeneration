"""
train_sedd_enhanced.py
-----------------------
Trains DiscreteDiffusionTransformer (SEDD) in two modes:

  --mode unconditional   : sequence = [token1, token2, ..., token4096]
  --mode conditional     : class embedding added (additive bias, as in SEDD.py)

Outputs → /data/joshi/tmp/MeshGeneration/runs/sedd_<mode>_<timestamp>/
  checkpoints/
  plots/          ← training curves, perplexity, code dist, per-class accuracy, entropy
  logs/train.log
  report.json

Usage:
  python train_sedd_enhanced.py --mode unconditional
  python train_sedd_enhanced.py --mode conditional
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
SRC_DIFFUSION = os.path.dirname(os.path.abspath(__file__))
SRC_MESHVQVAE = "/data/joshi/tmp/MeshGeneration/mesh_vqvae/src"
sys.path.insert(0, SRC_DIFFUSION)
sys.path.insert(0, SRC_MESHVQVAE)

from SEDD import DiscreteDiffusionTransformer
from preprocessing import MODELNET40_CLASSES

DATA_DIR = "/data/joshi/tmp/MeshGeneration/runs/sedd_data"
OUT_BASE  = "/data/joshi/tmp/MeshGeneration/runs"

# ── global plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#CCCCCC",
    "axes.linewidth":   0.8,
    "grid.color":       "#E5E5E5",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "text.color":       "#222222",
})
PALETTE = ["#5B8DB8", "#F4A35A", "#6DBF8A", "#D96B6B", "#A48CC4"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CodeSequenceDataset(Dataset):
    def __init__(self, pt_path: str, n_samples: int = -1):
        data = torch.load(pt_path, weights_only=False)
        codes  = data["codes"].long()
        labels = data["labels"].long()
        if n_samples > 0:
            idx    = torch.randperm(len(codes))[:n_samples]
            codes  = codes[idx]
            labels = labels[idx]
        self.codes  = codes
        self.labels = labels
        print(f"[Dataset] {os.path.basename(pt_path)}: {len(self.codes)} samples")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return {"input_ids": self.codes[idx], "class_labels": self.labels[idx]}


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Plot Callback
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedSEDDPlotCallback(Callback):
    """
    Generates the following plots every `plot_every` epochs:
      1. training_curves   — train/val loss + perplexity
      2. code_distribution — real vs generated code frequency
      3. per_class_gen     — generated code histograms for each class
      4. token_entropy     — entropy of generated distributions per class
      5. code_heatmap      — per-class code usage heatmap (classes × codebook)
    """

    def __init__(self, plot_dir: str, val_dataset, vocab_size: int,
                 seq_len: int, mode: str, plot_every: int = 5, n_gen: int = 8):
        super().__init__()
        self.plot_dir   = plot_dir
        self.val_ds     = val_dataset
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.mode       = mode          # "conditional" | "unconditional"
        self.plot_every = plot_every
        self.n_gen      = n_gen
        os.makedirs(plot_dir, exist_ok=True)

        # Tracked history
        self.train_losses  = []
        self.val_losses    = []
        self.perplexities  = []
        self.epochs        = []

        # Pre-compute real code histograms per class from val set
        self._real_hists = None

    def _build_real_hists(self):
        hists = {}
        for i in range(len(self.val_ds)):
            item = self.val_ds[i]
            c    = int(item["class_labels"])
            codes = item["input_ids"].numpy()
            if c not in hists:
                hists[c] = np.zeros(self.vocab_size, dtype=np.float64)
            hists[c] += np.bincount(codes, minlength=self.vocab_size)
        # normalize
        for c in hists:
            hists[c] = hists[c] / (hists[c].sum() + 1e-8)
        return hists

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        ep = trainer.current_epoch
        tl = float(metrics.get("train_loss", float("nan")))
        vl = float(metrics.get("val_loss",   float("nan")))

        # Perplexity = exp(loss)
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
                import traceback
                print(f"[PlotCallback] WARNING: {exc}\n{traceback.format_exc()}")

    # ── 1. Training curves ────────────────────────────────────────────────────

    def _plot_training_curves(self, ep: int):
        fig = plt.figure(figsize=(14, 5), constrained_layout=True)
        fig.suptitle(f"SEDD ({self.mode}) — Training Curves  [epoch {ep}]",
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

        # Loss
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.epochs, self.train_losses, color=PALETTE[0], lw=1.8,
                label="train_loss", alpha=0.9)
        ax.plot(self.epochs, self.val_losses,   color=PALETTE[1], lw=1.8,
                label="val_loss",   alpha=0.9)
        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_ylabel("Cross-Entropy Loss", labelpad=8)
        ax.set_title("Loss", fontsize=12, fontweight="semibold", pad=8)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

        # Perplexity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.epochs, self.perplexities, color=PALETTE[2], lw=1.8, alpha=0.9)
        ax2.set_xlabel("Epoch", labelpad=8)
        ax2.set_ylabel("Perplexity  (exp(val_loss))", labelpad=8)
        ax2.set_title("Perplexity", fontsize=12, fontweight="semibold", pad=8)

        fig.savefig(os.path.join(self.plot_dir, f"training_curves_ep{ep:04d}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ── 2. Real vs Generated code distribution ────────────────────────────────

    @torch.no_grad()
    def _plot_code_distribution(self, model, ep: int):
        device = next(model.parameters()).device

        # Real histogram (all val)
        real_h = np.zeros(self.vocab_size, dtype=np.float64)
        for i in range(len(self.val_ds)):
            real_h += np.bincount(self.val_ds[i]["input_ids"].numpy(),
                                  minlength=self.vocab_size)
        real_h /= real_h.sum() + 1e-8

        # Generated histogram
        n_gen = min(100, len(self.val_ds))
        cls_lbl = None
        if self.mode == "conditional" and model.use_class_condition:
            cls_lbl = torch.randint(0, model.num_classes, (n_gen,), device=device)
        gen = model.generate(batch_size=n_gen, seq_len=self.seq_len,
                             class_labels=cls_lbl, temperature=1.0, num_steps=50)
        gen_h = np.bincount(gen.cpu().numpy().flatten(),
                            minlength=self.vocab_size).astype(np.float64)
        gen_h /= gen_h.sum() + 1e-8

        fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
        x = np.arange(self.vocab_size)
        ax.bar(x, real_h, color=PALETTE[0], alpha=0.75, label="Real",      width=1.0)
        ax.bar(x, gen_h,  color=PALETTE[1], alpha=0.75, label="Generated", width=1.0)
        ax.set_xlabel("Code ID", labelpad=8)
        ax.set_ylabel("Frequency", labelpad=8)
        ax.set_title(f"Real vs Generated Code Distribution  [epoch {ep}]",
                     fontsize=13, fontweight="bold", pad=10)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)

        fig.savefig(os.path.join(self.plot_dir, f"code_dist_ep{ep:04d}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ── 3. Per-class generated histograms (first 10 classes) ─────────────────

    @torch.no_grad()
    def _plot_per_class_gen(self, model, ep: int):
        device   = next(model.parameters()).device
        n_cls    = min(10, model.num_classes or 10)
        n_cols   = 5
        n_rows   = (n_cls + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.5, n_rows * 3.0),
                                 constrained_layout=True)
        fig.suptitle(f"SEDD ({self.mode}) — Generated Code Histograms per Class  [epoch {ep}]",
                     fontsize=13, fontweight="bold")
        axes = np.array(axes).flatten()

        cls_lbl = None
        if self.mode == "conditional" and model.use_class_condition:
            cls_lbl = torch.arange(n_cls, device=device)

        samples = model.generate(batch_size=n_cls, seq_len=self.seq_len,
                                 class_labels=cls_lbl, temperature=1.0, num_steps=50)

        for i in range(n_cls):
            codes = samples[i].cpu().numpy()
            axes[i].bar(np.arange(self.vocab_size),
                        np.bincount(codes, minlength=self.vocab_size),
                        color=PALETTE[i % len(PALETTE)], alpha=0.85, width=1.0)
            cls_name = MODELNET40_CLASSES[i] if i < len(MODELNET40_CLASSES) else str(i)
            axes[i].set_title(cls_name, fontsize=10, fontweight="semibold")
            axes[i].set_xlabel("Code ID", fontsize=9, labelpad=6)
            axes[i].set_ylabel("Count",   fontsize=9, labelpad=6)

        for idx in range(n_cls, len(axes)):
            axes[idx].set_visible(False)

        fig.savefig(os.path.join(self.plot_dir, f"per_class_gen_ep{ep:04d}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ── 4. Token entropy per class ─────────────────────────────────────────────

    @torch.no_grad()
    def _plot_token_entropy(self, model, ep: int):
        device  = next(model.parameters()).device
        n_cls   = min(40, model.num_classes or 40)
        entropy_gen  = []
        entropy_real = []

        if self._real_hists is None:
            self._real_hists = self._build_real_hists()

        for c in range(n_cls):
            # Real entropy
            rh = self._real_hists.get(c, np.ones(self.vocab_size) / self.vocab_size)
            rh = rh + 1e-10
            entropy_real.append(-np.sum(rh * np.log(rh)))

            # Generated entropy
            cls_lbl = None
            if self.mode == "conditional" and model.use_class_condition:
                cls_lbl = torch.full((4,), c, dtype=torch.long, device=device)
            gen = model.generate(batch_size=4, seq_len=self.seq_len,
                                 class_labels=cls_lbl, temperature=1.0, num_steps=20)
            gh = np.bincount(gen.cpu().numpy().flatten(),
                             minlength=self.vocab_size).astype(np.float64)
            gh = gh / (gh.sum() + 1e-10)
            gh = gh + 1e-10
            entropy_gen.append(-np.sum(gh * np.log(gh)))

        cls_names = [MODELNET40_CLASSES[i] if i < len(MODELNET40_CLASSES) else str(i)
                     for i in range(n_cls)]
        x = np.arange(n_cls)

        fig, ax = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
        w = 0.35
        ax.bar(x - w/2, entropy_real, width=w, color=PALETTE[0], alpha=0.85,
               label="Real", zorder=3)
        ax.bar(x + w/2, entropy_gen,  width=w, color=PALETTE[1], alpha=0.85,
               label="Generated", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(cls_names, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Class", labelpad=8)
        ax.set_ylabel("Token Entropy (nats)", labelpad=8)
        ax.set_title(f"SEDD ({self.mode}) — Token Entropy per Class  [epoch {ep}]",
                     fontsize=13, fontweight="bold", pad=10)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
        ax.yaxis.grid(True, color="#E5E5E5", lw=0.6, zorder=0)

        fig.savefig(os.path.join(self.plot_dir, f"token_entropy_ep{ep:04d}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ── 5. Code usage heatmap (classes × codebook) ────────────────────────────

    @torch.no_grad()
    def _plot_code_heatmap(self, model, ep: int):
        device = next(model.parameters()).device
        n_cls  = min(20, model.num_classes or 40)

        mat = np.zeros((n_cls, self.vocab_size), dtype=np.float64)
        for c in range(n_cls):
            cls_lbl = None
            if self.mode == "conditional" and model.use_class_condition:
                cls_lbl = torch.full((4,), c, dtype=torch.long, device=device)
            gen = model.generate(batch_size=4, seq_len=self.seq_len,
                                 class_labels=cls_lbl, temperature=1.0, num_steps=20)
            h = np.bincount(gen.cpu().numpy().flatten(), minlength=self.vocab_size).astype(np.float64)
            mat[c] = h / (h.sum() + 1e-8)

        cls_names = [MODELNET40_CLASSES[i] if i < len(MODELNET40_CLASSES) else str(i)
                     for i in range(n_cls)]

        fig, ax = plt.subplots(figsize=(18, 5), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(np.arange(n_cls))
        ax.set_yticklabels(cls_names, fontsize=9)
        ax.set_xlabel("Code ID", labelpad=8)
        ax.set_ylabel("Class",   labelpad=8)
        ax.set_title(f"SEDD ({self.mode}) — Code Usage Heatmap (Generated)  [epoch {ep}]",
                     fontsize=13, fontweight="bold", pad=10)
        plt.colorbar(im, ax=ax, label="Frequency", shrink=0.8)

        fig.savefig(os.path.join(self.plot_dir, f"code_heatmap_ep{ep:04d}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model configs
# ─────────────────────────────────────────────────────────────────────────────

FULL_CFG = dict(
    vocab_size      = 256,
    max_seq_len     = 4096,
    d_model         = 512,
    nhead           = 8,
    num_layers      = 6,
    dim_feedforward = 2048,
    dropout         = 0.1,
    mask_id         = 256,
    num_timesteps   = 1000,
    schedule_type   = "cosine",
    learning_rate   = 1e-4,
    beta1           = 0.9,
    beta2           = 0.99,
    weight_decay    = 0.01,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def run(mode: str, run_dir: str, gpus: int = 8):
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, "plots")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir  = os.path.join(run_dir, "logs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # Redirect stdout to log file (rank 0 only — Lightning handles DDP output)
    print(f"\n{'='*65}")
    print(f"  SEDD Training  —  mode={mode}  gpus={gpus}")
    print(f"  run_dir = {run_dir}")
    print(f"{'='*65}\n")

    # ── mode-specific settings ─────────────────────────────────────────────
    is_conditional = (mode == "conditional")
    cfg = dict(FULL_CFG)
    cfg["num_classes"] = 40 if is_conditional else None

    batch_size  = 32   # per GPU
    max_epochs  = 200
    plot_every  = 10
    num_workers = 8

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = CodeSequenceDataset(os.path.join(DATA_DIR, "train_codes.pt"))
    val_ds   = CodeSequenceDataset(os.path.join(DATA_DIR, "val_codes.pt"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"batch/GPU: {batch_size} | epochs: {max_epochs}")

    # ── model ─────────────────────────────────────────────────────────────
    model = DiscreteDiffusionTransformer(**cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total_params/1e6:.2f}M  |  conditional={is_conditional}")

    # ── callbacks ─────────────────────────────────────────────────────────
    plot_cb = EnhancedSEDDPlotCallback(
        plot_dir=plot_dir, val_dataset=val_ds,
        vocab_size=cfg["vocab_size"], seq_len=cfg["max_seq_len"],
        mode=mode, plot_every=plot_every,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"sedd_{mode}-{{epoch:04d}}-{{val_loss:.4f}}",
        monitor="val_loss", mode="min",
        save_top_k=3, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", patience=25, mode="min", verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── trainer ───────────────────────────────────────────────────────────
    strategy = "ddp_find_unused_parameters_false" if gpus > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=gpus,
        strategy=strategy,
        precision="bf16",
        callbacks=[plot_cb, ckpt_cb, early_stop_cb, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=run_dir,
        gradient_clip_val=1.0,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    elapsed = time.time() - t0

    if trainer.global_rank != 0:
        return {}

    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min")

    # ── final eval plots ───────────────────────────────────────────────────
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"[INFO] Best checkpoint: {best_ckpt}")
        best_model = DiscreteDiffusionTransformer.load_from_checkpoint(
            best_ckpt, map_location="cuda:0", weights_only=False,
        )
        best_model.eval().to("cuda:0")
        ep_final = trainer.current_epoch
        plot_cb._plot_training_curves(ep_final)
        plot_cb._plot_code_distribution(best_model, ep_final)
        plot_cb._plot_per_class_gen(best_model, ep_final)
        plot_cb._plot_token_entropy(best_model, ep_final)
        plot_cb._plot_code_heatmap(best_model, ep_final)

    # ── save report ────────────────────────────────────────────────────────
    results = {
        "mode":           mode,
        "best_val_loss":  float(trainer.callback_metrics.get("val_loss", -1)),
        "total_params_M": round(total_params / 1e6, 3),
        "train_epochs":   trainer.current_epoch,
        "train_time_min": round(elapsed / 60, 1),
        "best_ckpt":      best_ckpt,
        "n_train":        len(train_ds),
        "n_val":          len(val_ds),
        "timestamp":      __import__("datetime").datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Report → {run_dir}/report.json")
    print(f"[DONE] Plots  → {plot_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["unconditional", "conditional"],
                        default="unconditional",
                        help="unconditional=[t1,t2,...] | conditional=class+tokens")
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()

    run_dir = os.path.join(OUT_BASE, f"sedd_{args.mode}")

    run(args.mode, run_dir, gpus=args.gpus)


if __name__ == "__main__":
    main()
