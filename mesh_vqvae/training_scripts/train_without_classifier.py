"""
train_final.py — MeshGPT WITHOUT CLASSIFIER (full plotting enabled)

Features:
  - CLASSIFIER HEAD DISABLED (pure VQ-VAE + fingerprint)
  - Per-mesh preprocessing timing (cached to metadata.log)
  - Model size, GPU count, throughput stats
  - Improved DashboardCallback: SVG individual plots + PNG dashboard
  - Reconstruction plots (multi-view scatter, GT vs Pred vs Error)
  - metadata.log written to output directory
  - --quick flag: 2-epoch smoke test on 32 samples

Usage:
  python train_final.py --quick              # smoke test (~2 min)
  python train_final.py --gpus 8            # full run
  python train_final.py --gpus 8 --resume last
"""

import os, sys, time, json, glob, argparse, random, traceback
from datetime import datetime, timedelta
from textwrap import wrap as twrap

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

# ── resolve src path ─────────────────────────────────────────────────────────
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "new_implementation", "src")
sys.path.insert(0, os.path.abspath(SRC))

from config import SmallModelConfig, LargeModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset, build_dataloaders
from preprocessing import MODELNET40_CLASSES, load_or_compute

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#CCCCCC",
    "axes.linewidth":   0.8,
    "grid.color":       "#E5E5E5",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "text.color":       "#222222",
})

PALETTE = ["#5B8DB8", "#F4A35A", "#6DBF8A", "#D96B6B", "#A48CC4",
           "#E8C56D", "#7EB5A6", "#C47AB3"]

# ── output dirs ──────────────────────────────────────────────────────────────
OUT_DIR   = "/data/joshi/MESHGPT/without_clssifire"
DATA_DIR  = "/data/joshi/modelnet40_meshes"
CACHE_DIR = os.path.join(OUT_DIR, "cache")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TIMING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class Timer:
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def fmt(self):
        s = self.elapsed
        if s < 60:
            return f"{s:.2f}s"
        return str(timedelta(seconds=int(s)))


def time_preprocessing(data_dir, cache_dir, n_sample=20):
    """Time preprocessing for n_sample meshes, return stats dict."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.ply")))
    if not files:
        return {}
    files = files[:n_sample]
    times = []
    cache_hits = 0
    for f in files:
        t0 = time.perf_counter()
        data = load_or_compute(f, cache_dir, num_surface=2048, num_query=2048)
        dt = time.perf_counter() - t0
        times.append(dt)
        # If it was very fast (<0.05s) it was a cache hit
        if dt < 0.05:
            cache_hits += 1
    return {
        "n_sampled": len(times),
        "cache_hits": cache_hits,
        "avg_ms": float(np.mean(times)) * 1000,
        "min_ms": float(np.min(times)) * 1000,
        "max_ms": float(np.max(times)) * 1000,
        "std_ms": float(np.std(times)) * 1000,
    }


def model_size_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    return {
        "total_params": total,
        "trainable_params": trainable,
        "size_mb": round(size_mb, 2),
        "total_params_M": round(total / 1e6, 3),
    }


def gpu_stats():
    if not torch.cuda.is_available():
        return {}
    info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info[f"gpu_{i}"] = {
            "name": props.name,
            "memory_gb": round(props.total_memory / 1024**3, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        }
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path_no_ext, svg=True, dpi=150):
    """Save as PNG and optionally SVG."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    fig.savefig(path_no_ext + ".png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    if svg:
        try:
            fig.savefig(path_no_ext + ".svg", bbox_inches="tight",
                        facecolor="white", edgecolor="none")
        except Exception:
            pass  # SVG not supported for rasterized scatter — skip gracefully
    plt.close(fig)


def plot_loss_curves(hist, out_dir, epoch):
    """Individual SVG: reconstruction + VQ loss."""
    e = hist["epoch"]
    if len(e) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(f"Loss Curves — epoch {epoch}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(e, hist["recon"], color=PALETTE[0], linewidth=1.8, label="recon loss", alpha=0.9)
    ax.plot(e, hist["vq"],    color=PALETTE[1], linewidth=1.8, label="vq loss",    alpha=0.9)
    ax.set_xlabel("Epoch", labelpad=8); ax.set_ylabel("Loss", labelpad=8)
    ax.set_title("Reconstruction & VQ Loss", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    ax = axes[1]
    ax.plot(e, hist["cls_loss"], color=PALETTE[3], linewidth=1.8, label="cls loss",  alpha=0.9)
    ax.plot(e, hist["fp"],       color=PALETTE[4], linewidth=1.8, label="fp loss",   alpha=0.9)
    ax.set_xlabel("Epoch", labelpad=8); ax.set_ylabel("Loss", labelpad=8)
    ax.set_title("Classifier & Fingerprint Loss", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    _save(fig, os.path.join(out_dir, "loss_curves"), svg=True)


def plot_iou_history(hist, out_dir, epoch):
    """Individual SVG: IoU + cls_acc over epochs."""
    e = hist["epoch"]
    if len(e) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(f"Val Metrics Over Time — epoch {epoch}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(e, hist["iou"], color=PALETTE[2], linewidth=2.2, label="val IoU", alpha=0.95)
    ax.axhline(0.38, color="#D96B6B", linestyle="--", alpha=0.7, label="target 0.38")
    ax.fill_between(e, hist["iou"], alpha=0.12, color=PALETTE[2])
    ax.set_xlabel("Epoch", labelpad=8); ax.set_ylabel("IoU", labelpad=8)
    ax.set_title("Val IoU Over Time", fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 1.0); ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    ax = axes[1]
    cls_pct = np.array(hist["cls_acc"]) * 100
    ax.plot(e, cls_pct, color=PALETTE[0], linewidth=2.2, label="cls acc %", alpha=0.95)
    ax.axhline(2.5, color="#888", linestyle=":", alpha=0.6, label="random 2.5%")
    ax.fill_between(e, cls_pct, alpha=0.12, color=PALETTE[0])
    ax.set_xlabel("Epoch", labelpad=8); ax.set_ylabel("Accuracy (%)", labelpad=8)
    ax.set_title("Classification Accuracy %", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    _save(fig, os.path.join(out_dir, "iou_history"), svg=True)


def plot_codebook(counts, K, out_dir, epoch, util, perp):
    """Individual SVG: codebook usage histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    fig.suptitle(f"Codebook Health — epoch {epoch} | util={util*100:.1f}%  perp={perp:.1f}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    dead = (counts == 0).sum()
    ax.bar(np.arange(K), np.log1p(counts), width=1.0, linewidth=0,
           color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Code Index", labelpad=8); ax.set_ylabel("log(1+count)", labelpad=8)
    ax.set_title(f"\n".join(twrap(
        f"Code Usage (log scale) — {dead} dead / {K} total", 55)),
        fontsize=11, fontweight="semibold")

    ax = axes[1]
    freq = counts / (counts.sum() + 1e-8)
    ax.bar(np.arange(K), freq, width=1.0, linewidth=0, color=PALETTE[1], alpha=0.85)
    ax.set_xlabel("Code Index", labelpad=8); ax.set_ylabel("Frequency", labelpad=8)
    ax.set_title("Code Usage Frequency (normalized)", fontsize=11, fontweight="semibold")

    _save(fig, os.path.join(out_dir, "codebook"), svg=True)


def plot_iou_distribution(ious, out_dir, epoch):
    """Individual SVG: IoU distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(ious, bins=25, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(ious.mean(), color=PALETTE[3], linestyle="--", linewidth=1.8,
               label=f"mean={ious.mean():.3f}")
    ax.axvline(np.median(ious), color=PALETTE[1], linestyle=":", linewidth=1.8,
               label=f"median={np.median(ious):.3f}")
    ax.axvline(0.38, color="#888", linestyle=":", alpha=0.6, label="target 0.38")
    ax.set_xlabel("IoU", labelpad=8); ax.set_ylabel("Count", labelpad=8)
    ax.set_title(f"IoU Distribution — epoch {epoch}", fontsize=12, fontweight="semibold")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    _save(fig, os.path.join(out_dir, "iou_distribution"), svg=True)


def plot_per_class_iou(ious, labels_arr, out_dir, epoch):
    """Individual SVG: per-class IoU bar chart."""
    unique = sorted(np.unique(labels_arr))
    cls_iou = [ious[labels_arr == l].mean() if (labels_arr == l).any() else 0.0
               for l in unique]
    names   = [MODELNET40_CLASSES[l][:10] for l in unique]
    order   = np.argsort(cls_iou)[::-1]

    n = len(order)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 5), constrained_layout=True)
    colors = [PALETTE[2] if cls_iou[i] > 0.40 else
              (PALETTE[1] if cls_iou[i] > 0.25 else PALETTE[3]) for i in order]
    ax.barh([names[i] for i in order], [cls_iou[i] for i in order],
            color=colors, alpha=0.85)
    ax.axvline(np.mean(cls_iou), color="navy", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(cls_iou):.3f}")
    ax.set_xlabel("IoU", labelpad=8)
    ax.set_title(f"Per-Class IoU — epoch {epoch}\n"
                 f"(green>0.40 / orange>0.25 / red<0.25)",
                 fontsize=11, fontweight="semibold")
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    _save(fig, os.path.join(out_dir, "per_class_iou"), svg=True)


def plot_reconstruction(recon_ex, out_dir, epoch):
    """PNG (no SVG — rasterized scatter): GT vs Pred vs Error multi-view."""
    n = len(recon_ex)
    if n == 0:
        return
    # 3 columns per object: XY, XZ, YZ — then Error-YZ
    n_cols = 7  # GT-XY Pred-XY GT-XZ Pred-XZ GT-YZ Pred-YZ Error
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 2.6, n * 2.6),
                             constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    fig.suptitle(
        f"Reconstruction — epoch {epoch}  "
        f"(GT=blue | Pred=orange | Error=heatmap)",
        fontsize=12, fontweight="bold"
    )

    col_labels = ["GT XY", "Pred XY", "GT XZ", "Pred XZ", "GT YZ", "Pred YZ", "Error YZ"]

    for row, d in enumerate(recon_ex):
        pts       = d["pts"]          # [M, 3]
        gt        = d["gt"]           # [M] float
        pred_prob = d["pred"]         # [M] float 0..1
        pred_bin  = (pred_prob > 0.5).astype(float)
        err       = np.abs(pred_bin - gt)

        in_gt   = pts[gt > 0.5]
        in_pred = pts[pred_bin > 0.5]

        views = [
            (in_gt,   0, 1, PALETTE[0]),
            (in_pred, 0, 1, PALETTE[1]),
            (in_gt,   0, 2, PALETTE[0]),
            (in_pred, 0, 2, PALETTE[1]),
            (in_gt,   1, 2, PALETTE[0]),
            (in_pred, 1, 2, PALETTE[1]),
        ]
        for col, (v_pts, xi, yi, col_c) in enumerate(views):
            ax = axes[row, col]
            if len(v_pts):
                ax.scatter(v_pts[:, xi], v_pts[:, yi],
                           c=col_c, s=1.5, alpha=0.4,
                           linewidths=0, rasterized=True)
            lbl = d["label"].upper() if col == 0 else ""
            iou_str = f"\nIoU={d['iou']:.3f}" if col == 1 else ""
            ax.set_title(f"{lbl}{col_labels[col]}{iou_str}", fontsize=6, pad=2)
            ax.set_aspect("equal"); ax.axis("off")

        # Error column
        ax = axes[row, 6]
        sc = ax.scatter(pts[:, 1], pts[:, 2], c=err,
                        cmap="RdYlGn_r", s=1.5, alpha=0.5,
                        vmin=0, vmax=1, linewidths=0, rasterized=True)
        ax.set_title(f"Error YZ\n|pred-gt|", fontsize=6, pad=2)
        ax.set_aspect("equal"); ax.axis("off")
        if row == 0:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # No SVG for this one (rasterized)
    fig.savefig(os.path.join(out_dir, "reconstruction.png"),
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tsne(fps_arr, labels_arr, out_dir, epoch):
    """Individual PNG/SVG: t-SNE of fingerprints."""
    try:
        from sklearn.manifold import TSNE
        if len(fps_arr) < 5:
            return
        perp = min(30, len(fps_arr) - 1)
        emb = TSNE(n_components=2, random_state=42,
                   perplexity=perp, max_iter=500).fit_transform(fps_arr)
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        cmap = matplotlib.colormaps.get_cmap("tab20")
        for lbl in np.unique(labels_arr):
            m = labels_arr == lbl
            ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(lbl % 20)],
                       label=MODELNET40_CLASSES[lbl], s=18, alpha=0.85,
                       linewidths=0)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, ncol=2, frameon=True,
                  framealpha=0.9, edgecolor="#CCCCCC")
        ax.set_title(f"Fingerprint t-SNE — epoch {epoch}",
                     fontsize=12, fontweight="semibold")
        ax.set_xlabel("t-SNE dim 1", labelpad=8)
        ax.set_ylabel("t-SNE dim 2", labelpad=8)
        _save(fig, os.path.join(out_dir, "tsne"), svg=True)
    except Exception as ex:
        print(f"[Dashboard] t-SNE skipped: {ex}")


def plot_dashboard(hist, data, epoch, step, out_dir):
    """Master 3×4 PNG dashboard combining all metrics."""
    ious       = data["ious"]
    cls_arr    = data["cls_arr"]
    counts     = data["counts"]
    util       = data["util"]
    perp       = data["perp"]
    labels_arr = data["labels_arr"]
    K          = data["K"]
    e          = hist["epoch"]

    fig = plt.figure(figsize=(22, 15), facecolor="white")
    cls_pct_str = "n/a" if len(cls_arr) == 0 else f"{cls_arr.mean()*100:.1f}%"
    fig.suptitle(
        f"MeshGPT Training Dashboard  |  epoch={epoch}  step={step}\n"
        f"IoU={ious.mean():.4f}   Cls={cls_pct_str}   "
        f"Util={util*100:.1f}%   Perp={perp:.1f}",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

    # ── row 0: loss curves ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    if len(e) >= 2:
        ax.plot(e, hist["recon"], color=PALETTE[0], linewidth=1.8,
                label="recon", alpha=0.9)
        ax.plot(e, hist["vq"],   color=PALETTE[1], linewidth=1.8,
                label="vq", alpha=0.8)
    ax.set_title("Reconstruction & VQ Loss", fontweight="semibold")
    ax.set_xlabel("Epoch", labelpad=6); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2:])
    if len(e) >= 2:
        ax.plot(e, hist["cls_loss"], color=PALETTE[3], linewidth=1.8,
                label="cls", alpha=0.9)
        ax.plot(e, hist["fp"],       color=PALETTE[4], linewidth=1.8,
                label="fp",  alpha=0.8)
    ax.set_title("Classifier & Fingerprint Loss", fontweight="semibold")
    ax.set_xlabel("Epoch", labelpad=6); ax.legend(fontsize=8)

    # ── row 1: IoU history, cls acc, IoU hist, codebook ───────────────────
    ax = fig.add_subplot(gs[1, 0])
    if len(e) >= 2:
        ax.plot(e, hist["iou"], color=PALETTE[2], linewidth=2.0)
        ax.fill_between(e, hist["iou"], alpha=0.12, color=PALETTE[2])
        ax.axhline(0.38, color=PALETTE[3], linestyle="--",
                   alpha=0.7, label="target")
    ax.set_title("Val IoU Over Time", fontweight="semibold")
    ax.set_xlabel("Epoch", labelpad=6); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    if len(e) >= 2:
        cls_pct = np.array(hist["cls_acc"]) * 100
        ax.plot(e, cls_pct, color=PALETTE[0], linewidth=2.0)
        ax.fill_between(e, cls_pct, alpha=0.12, color=PALETTE[0])
        ax.axhline(2.5, color="#888", linestyle=":", alpha=0.5, label="random")
    ax.set_title("Cls Accuracy %", fontweight="semibold")
    ax.set_xlabel("Epoch", labelpad=6); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(ious, bins=22, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(ious.mean(), color=PALETTE[3], linestyle="--", linewidth=1.8,
               label=f"μ={ious.mean():.3f}")
    ax.axvline(0.38, color="#888", linestyle=":", alpha=0.6, label="target")
    ax.set_title("IoU Distribution", fontweight="semibold")
    ax.set_xlabel("IoU", labelpad=6); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 3])
    dead = (counts == 0).sum()
    ax.bar(np.arange(K), np.log1p(counts), width=1.0, linewidth=0,
           color=PALETTE[1], alpha=0.85)
    ax.set_title(f"Codebook (log) — {dead} dead/{K}",
                 fontweight="semibold")
    ax.set_xlabel("Code index", labelpad=6)

    # ── row 2: per-class IoU ──────────────────────────────────────────────
    unique = sorted(np.unique(labels_arr))
    cv     = [ious[labels_arr == l].mean() if (labels_arr == l).any() else 0.0
              for l in unique]
    names  = [MODELNET40_CLASSES[l][:8] for l in unique]
    order  = np.argsort(cv)[::-1]
    colors = [PALETTE[2] if cv[i] > 0.40 else
              (PALETTE[1] if cv[i] > 0.25 else PALETTE[3]) for i in order]
    ax = fig.add_subplot(gs[2, :])
    ax.barh([names[i] for i in order], [cv[i] for i in order],
            color=colors, alpha=0.85)
    ax.axvline(ious.mean(), color="navy", linestyle="--", linewidth=1.4,
               label=f"mean={ious.mean():.3f}")
    ax.set_title("Per-Class IoU  (green>0.40 | orange>0.25 | red<0.25)",
                 fontweight="semibold")
    ax.set_xlabel("IoU", labelpad=6); ax.set_xlim(0, 1.0)
    ax.legend(fontsize=8)

    path = os.path.join(out_dir, "dashboard")
    fig.savefig(path + ".png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DASHBOARD CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardCallback(Callback):
    """
    Every `every_n_val` validation epochs:
      - Collects 200 val samples
      - Saves individual SVG plots + PNG dashboard
      - Saves reconstruction.png
      - Updates metadata.log timing
    """

    def __init__(self, cfg, run_dir, every_n_val=1, meta_path=None):
        super().__init__()
        self.cfg       = cfg
        self.run_dir   = run_dir
        self.every_n   = every_n_val
        self.meta_path = meta_path
        self._val_count = 0
        self._metric_history = {k: [] for k in
            ["epoch", "iou", "cls_acc", "recon", "vq",
             "cls_loss", "fp", "codebook_util", "perplexity"]}
        self._train_start = time.perf_counter()
        self._epoch_times = []

    @torch.no_grad()
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.perf_counter()

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(self, "_epoch_t0"):
            self._epoch_times.append(time.perf_counter() - self._epoch_t0)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        self._val_count += 1
        if self._val_count % self.every_n != 0:
            return
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        step  = trainer.global_step
        print(f"\n[Dashboard] epoch={epoch} step={step} — collecting samples...")
        t0 = time.perf_counter()
        try:
            self._generate(pl_module, trainer, epoch, step)
        except Exception as e:
            print(f"[Dashboard] WARNING: {e}\n{traceback.format_exc()}")
        print(f"[Dashboard] done in {time.perf_counter()-t0:.1f}s")

    def _collect(self, model, device, n=200, seed=42, epoch=0):
        cfg = self.cfg
        ds  = ModelNet40Dataset(
            cfg.train.data_dir, cfg.train.cache_dir, split="val",
            num_surface=cfg.train.num_surface_points,
            num_query=cfg.train.num_query_points,
            use_augmentation=False, use_contrastive=False,
        )
        rng  = np.random.RandomState(seed + epoch * 997)
        idxs = rng.choice(len(ds), min(n, len(ds)), replace=False)
        K    = model.quantizer.num_embeddings
        use_cls = getattr(model, "use_classifier", True)

        ious, cls_ok, codes_list, fps_list, labels_list, recon_ex = [], [], [], [], [], []

        model.eval()
        with torch.no_grad():
            for i in idxs:
                try:
                    item = ds[int(i)]
                    b = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in item.items()}
                    b["label"] = torch.tensor([item["label"]], device=device)

                    out  = model(b["points"], b["normals"], b["curvature"],
                                 b["query_pts"], b["label"])
                    occ  = b["occupancy"]
                    prob = out["logits"].sigmoid()
                    pred = (prob > 0.5).float()
                    inter = (pred * occ).sum(1)
                    union = ((pred + occ) > 0).float().sum(1)
                    iou   = (inter / (union + 1e-8)).mean().item()

                    ious.append(iou)
                    labels_list.append(item["label"])

                    # Classifier accuracy (0 when disabled)
                    if use_cls and out.get("class_logits") is not None:
                        cls_ok.append(int(out["class_logits"].argmax(1).item() == item["label"]))
                    else:
                        cls_ok.append(0)

                    # Code indices — always available regardless of classifier
                    codes_list.append(out["code_indices"].cpu().squeeze(0))

                    # Fingerprint
                    fps_list.append(out["fingerprint"].cpu().squeeze(0))

                    # Reconstruction examples (up to 8)
                    if len(recon_ex) < 8:
                        recon_ex.append({
                            "label": MODELNET40_CLASSES[item["label"]],
                            "pts":   b["query_pts"].cpu().squeeze(0).numpy(),
                            "gt":    occ.cpu().squeeze(0).numpy(),
                            "pred":  prob.cpu().squeeze(0).numpy(),
                            "iou":   iou,
                        })
                except Exception as e:
                    print(f"[Dashboard] Sample {i} skipped: {e}")

        if not ious:
            return None

        ious       = np.array(ious)
        cls_arr    = np.array(cls_ok)
        labels_arr = np.array(labels_list)

        # Codebook usage
        codes_flat = torch.stack(codes_list).reshape(-1).numpy()
        counts     = np.bincount(codes_flat, minlength=K)
        util       = float((counts > 0).sum()) / K
        perp       = float(np.exp(-np.sum(
            (counts / counts.sum() + 1e-8) *
            np.log(counts / counts.sum() + 1e-8))))

        fps_arr = torch.stack(fps_list).numpy()

        return dict(ious=ious, cls_arr=cls_arr, counts=counts, util=util,
                    perp=perp, labels_arr=labels_arr, fps_arr=fps_arr,
                    recon_ex=recon_ex, K=K)

    def _generate(self, model, trainer, epoch, step):
        model.eval()
        device = next(model.parameters()).device

        data = self._collect(model, device, n=200, seed=42, epoch=epoch)
        if data is None:
            model.train(); return

        ious    = data["ious"]
        cls_arr = data["cls_arr"]
        util    = data["util"]
        perp    = data["perp"]
        counts  = data["counts"]
        labels_arr = data["labels_arr"]

        # Update history
        logged = trainer.logged_metrics
        self._metric_history["epoch"].append(epoch)
        self._metric_history["iou"].append(float(ious.mean()))
        self._metric_history["cls_acc"].append(float(cls_arr.mean()) if len(cls_arr) > 0 else 0.0)
        self._metric_history["codebook_util"].append(util)
        self._metric_history["perplexity"].append(perp)
        self._metric_history["recon"].append(float(logged.get("train/recon", 0)))
        self._metric_history["vq"].append(float(logged.get("train/vq", 0)))
        self._metric_history["cls_loss"].append(float(logged.get("train/cls", 0)))
        self._metric_history["fp"].append(float(logged.get("train/fp", 0)))

        tag     = f"ep{epoch:04d}"
        edir    = os.path.join(self.run_dir, "eval", tag)
        os.makedirs(edir, exist_ok=True)

        # Individual plots
        plot_loss_curves(self._metric_history, edir, epoch)
        plot_iou_history(self._metric_history, edir, epoch)
        plot_codebook(counts, data["K"], edir, epoch, util, perp)
        plot_iou_distribution(ious, edir, epoch)
        plot_per_class_iou(ious, labels_arr, edir, epoch)
        plot_reconstruction(data["recon_ex"], edir, epoch)
        plot_tsne(data["fps_arr"], labels_arr, edir, epoch)

        # Master dashboard PNG
        plot_dashboard(self._metric_history, data, epoch, step, edir)

        # Also copy latest plots to top-level plots/ for quick browsing
        latest_dir = os.path.join(self.run_dir, "plots", "latest")
        os.makedirs(latest_dir, exist_ok=True)
        import shutil
        for fname in os.listdir(edir):
            src = os.path.join(edir, fname)
            dst = os.path.join(latest_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # Append timing to metadata
        if self.meta_path and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path) as f:
                    meta = json.load(f)
                elapsed_h = (time.perf_counter() - self._train_start) / 3600
                meta["training"]["elapsed_hours"] = round(elapsed_h, 3)
                meta["training"]["current_epoch"] = epoch
                meta["training"]["current_iou"]   = round(float(ious.mean()), 4)
                meta["training"]["current_cls_acc"] = round(float(cls_arr.mean()) if len(cls_arr) > 0 else 0.0, 4)
                if self._epoch_times:
                    meta["training"]["avg_epoch_seconds"] = round(
                        float(np.mean(self._epoch_times[-10:])), 1)
                with open(self.meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        dead = int((counts == 0).sum())
        cls_mean = float(cls_arr.mean()) if len(cls_arr) > 0 and getattr(model, "use_classifier", True) else float("nan")
        print(f"[Dashboard] IoU={ious.mean():.4f}  util={util:.3f}  "
              f"cls={'n/a' if cls_mean!=cls_mean else f'{cls_mean:.3f}'}  dead={dead}  → {edir}/")
        model.train()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. METADATA
# ═══════════════════════════════════════════════════════════════════════════════

def write_metadata(out_dir, cfg, model, args, preproc_timing):
    meta = {
        "run": {
            "timestamp":   datetime.now().isoformat(),
            "output_dir":  out_dir,
            "model_type":  args.model,
            "gpus":        args.gpus,
            "quick_test":  args.quick,
        },
        "model": model_size_stats(model),
        "config": {
            "grid_res":        cfg.encoder.grid_res,
            "codebook_size":   cfg.vq.num_embeddings,
            "embedding_dim":   cfg.vq.embedding_dim,
            "topk_ratio":      cfg.masker.topk_ratio,
            "batch_size":      cfg.train.batch_size,
            "lr":              cfg.train.lr,
            "pos_weight":      4.0,
        },
        "gpus": gpu_stats(),
        "preprocessing": preproc_timing,
        "training": {
            "elapsed_hours":     0.0,
            "current_epoch":     0,
            "current_iou":       0.0,
            "current_cls_acc":   0.0,
            "avg_epoch_seconds": 0.0,
        },
        "data": {
            "data_dir":  DATA_DIR,
            "cache_dir": CACHE_DIR,
        },
    }
    meta_path = os.path.join(out_dir, "metadata.log")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Metadata] Written → {meta_path}")
    print(f"[Metadata] Model: {meta['model']['total_params_M']}M params  "
          f"({meta['model']['size_mb']} MB)")
    if preproc_timing:
        print(f"[Metadata] Preprocessing: avg={preproc_timing['avg_ms']:.1f}ms/mesh  "
              f"cache_hits={preproc_timing['cache_hits']}/{preproc_timing['n_sampled']}")
    return meta_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ARGS + MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="small", choices=["small", "large"])
    p.add_argument("--gpus",   type=int, default=8)
    p.add_argument("--resume", default=None)
    p.add_argument("--quick",  action="store_true",
                   help="Smoke test: 2 epochs, 32 samples, 1 GPU")
    return p.parse_args()


def main():
    args = parse_args()

    # ── config ───────────────────────────────────────────────────────────
    cfg = SmallModelConfig() if args.model == "small" else LargeModelConfig()
    cfg.train.use_classifier = False  # DISABLE CLASSIFIER HEAD
    cfg.train.data_dir   = DATA_DIR
    cfg.train.cache_dir  = CACHE_DIR
    cfg.train.output_dir = OUT_DIR
    cfg.train.num_gpus   = 1 if args.quick else args.gpus

    for d in [OUT_DIR, CACHE_DIR,
              os.path.join(OUT_DIR, "checkpoints"),
              os.path.join(OUT_DIR, "plots"),
              os.path.join(OUT_DIR, "logs"),
              os.path.join(OUT_DIR, "eval")]:
        os.makedirs(d, exist_ok=True)

    pl.seed_everything(cfg.train.seed)

    # ── time preprocessing ───────────────────────────────────────────────
    print("\n[Preprocessing] Timing mesh preprocessing...")
    with Timer("preproc") as t_pre:
        preproc_timing = time_preprocessing(DATA_DIR, CACHE_DIR, n_sample=30)
    print(f"[Preprocessing] Done in {t_pre.fmt()}")

    # ── build data ───────────────────────────────────────────────────────
    if args.quick:
        # Small subset for smoke test
        full_ds = ModelNet40Dataset(
            DATA_DIR, CACHE_DIR, split="train",
            num_surface=2048, num_query=2048,
            use_augmentation=True, use_contrastive=False,
        )
        val_ds = ModelNet40Dataset(
            DATA_DIR, CACHE_DIR, split="val",
            num_surface=2048, num_query=2048,
            use_augmentation=False, use_contrastive=False,
        )
        idxs = np.random.choice(len(full_ds), min(32, len(full_ds)), replace=False)
        train_ds = Subset(full_ds, idxs.tolist())
        val_ds   = Subset(val_ds, list(range(min(16, len(val_ds)))))
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False,
                                  num_workers=2, pin_memory=True)
        cfg.train.batch_size = 4
        max_epochs = 2
        patience   = 5
    else:
        train_loader, val_loader = build_dataloaders(cfg.train, num_gpus=args.gpus)
        max_epochs = 1000
        patience   = 30

    # ── model ────────────────────────────────────────────────────────────
    model = MaskedVQVAE3D(cfg)
    print(f"[Model] {args.model.upper()} — "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # ── write initial metadata ───────────────────────────────────────────
    meta_path = write_metadata(OUT_DIR, cfg, model, args, preproc_timing)

    # ── callbacks ────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(OUT_DIR, "checkpoints"),
        filename=f"{args.model}-{{epoch:04d}}-{{val/iou:.4f}}-{{val/cls_acc:.4f}}",
        monitor="val/iou",
        mode="max",
        save_top_k=5,
        save_last=True,
    )
    lr_cb    = LearningRateMonitor(logging_interval="step")
    dash_cb  = DashboardCallback(cfg, OUT_DIR, every_n_val=1, meta_path=meta_path)
    early_cb = EarlyStopping(monitor="val/iou", mode="max",
                             patience=patience, min_delta=0.001, verbose=True)

    logger = CSVLogger(save_dir=os.path.join(OUT_DIR, "logs"), name="csv", version=0)

    from pytorch_lightning.strategies import DDPStrategy

    n_gpus    = 1 if args.quick else args.gpus
    strategy  = ("auto" if n_gpus == 1
                 else DDPStrategy(find_unused_parameters=True))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=n_gpus,
        strategy=strategy,
        precision="bf16",
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=max(1, min(cfg.train.log_every, 10)),
        check_val_every_n_epoch=1,
        callbacks=[ckpt_cb, lr_cb, dash_cb, early_cb],
        logger=logger,
        enable_progress_bar=True,
        default_root_dir=OUT_DIR,
    )

    # ── train ────────────────────────────────────────────────────────────
    t_train_start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"  MeshGPT WITHOUT CLASSIFIER — {'QUICK TEST' if args.quick else 'FULL TRAINING'}")
    print(f"  GPUs: {n_gpus}  |  Output: {OUT_DIR}")
    print(f"{'='*60}\n")

    try:
        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=val_loader, ckpt_path=args.resume)
    except RuntimeError as e:
        if "Missing folder" in str(e):
            pass  # CSV logger cleanup artifact — safe to ignore
        else:
            raise

    elapsed = time.perf_counter() - t_train_start

    # ── final metadata update ────────────────────────────────────────────
    if trainer.global_rank == 0:
        with open(meta_path) as f:
            meta = json.load(f)
        meta["training"]["total_seconds"]    = round(elapsed, 1)
        meta["training"]["total_hours"]      = round(elapsed / 3600, 3)
        meta["training"]["final_epoch"]      = trainer.current_epoch
        meta["training"]["status"]           = "complete"
        if dash_cb._epoch_times:
            avg_ep = float(np.mean(dash_cb._epoch_times))
            meta["training"]["avg_epoch_seconds"] = round(avg_ep, 1)
            meta["training"]["estimated_full_run_hours"] = round(
                avg_ep * 1000 / 3600, 1)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n[DONE] Training complete in {timedelta(seconds=int(elapsed))}")
        print(f"[DONE] Checkpoints → {OUT_DIR}/checkpoints/")
        print(f"[DONE] Plots       → {OUT_DIR}/plots/")
        print(f"[DONE] Metadata    → {meta_path}")


if __name__ == "__main__":
    main()
