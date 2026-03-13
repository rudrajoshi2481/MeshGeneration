"""
visualize.py — UMAP plots, codebook utilization, fingerprint retrieval.
Usage:
    python visualize.py --checkpoint <path.ckpt>   # from saved checkpoint
    python visualize.py --random                    # random-init model (smoke test)

Output → /data/joshi/MESHGPT/new_implementation/trash/plots/
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(__file__))

from config import SmallModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset
from preprocessing import MODELNET40_CLASSES

OUTPUT_DIR = "/data/joshi/MESHGPT/new_implementation/trash/plots"
CACHE_DIR  = "/data/joshi/MESHGPT/new_implementation/trash/cache"
DATA_DIR   = "/data/joshi/modelnet40_meshes"
DEVICE     = "cuda:0"
N_SAMPLES  = 200   # samples to embed


@torch.no_grad()
def collect_embeddings(model, ds, n=N_SAMPLES, device=DEVICE):
    """Run model on n samples, collect fingerprints + code indices + labels."""
    model.eval()
    all_fps, all_labels, all_codes = [], [], []
    all_util = []

    indices = np.random.choice(len(ds), min(n, len(ds)), replace=False)
    for i in indices:
        item = ds[i]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in item.items()}
        batch["label"] = batch["label"].unsqueeze(0) if batch["label"].dim() == 0 else batch["label"]

        try:
            out = model(
                batch["points"], batch["normals"], batch["curvature"],
                batch["query_pts"], batch["label"],
            )
            all_fps.append(out["fingerprint"].cpu().squeeze(0))
            all_labels.append(item["label"].item())
            all_codes.append(out["code_indices"].cpu().squeeze(0))
        except Exception as e:
            print(f"  skip sample {i}: {e}")
            continue

    fps = torch.stack(all_fps).numpy()        # [n, fp_dim]
    labels = np.array(all_labels)             # [n]
    codes = torch.stack(all_codes)            # [n, N_vox]
    return fps, labels, codes


def plot_umap_fingerprints(fps, labels, output_dir):
    """UMAP of fingerprint embeddings colored by class."""
    try:
        import umap
    except ImportError:
        print("[Vis] umap-learn not available, skipping UMAP")
        return

    print("[Vis] Running UMAP on fingerprints...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(fps)  # [n, 2]

    n_classes = len(MODELNET40_CLASSES)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[cmap(lbl)], label=MODELNET40_CLASSES[lbl],
                   s=20, alpha=0.8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
              markerscale=2, ncol=2)
    ax.set_title("UMAP of Fingerprint Embeddings (colored by class)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    path = os.path.join(output_dir, "umap_fingerprints.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Vis] UMAP fingerprints → {path}")


def plot_umap_codebook(codebook_weights, output_dir):
    """UMAP of codebook vectors."""
    try:
        import umap
    except ImportError:
        return
    K, D = codebook_weights.shape
    if K > 2000:
        idx = np.random.choice(K, 2000, replace=False)
        weights = codebook_weights[idx]
        labels_cb = idx
    else:
        weights = codebook_weights
        labels_cb = np.arange(K)

    print("[Vis] Running UMAP on codebook vectors...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(weights)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels_cb, cmap="viridis", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Code Index")
    ax.set_title(f"UMAP of Codebook Vectors ({K} codes, D={D})")
    plt.tight_layout()
    path = os.path.join(output_dir, "umap_codebook.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Vis] UMAP codebook → {path}")


def plot_codebook_utilization(codes, K, labels, output_dir):
    """Codebook usage histogram, overall and per class."""
    all_codes_flat = codes.reshape(-1).numpy()
    counts = np.bincount(all_codes_flat, minlength=K)
    active = (counts > 0).sum()
    util = active / K

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Overall histogram
    axes[0].bar(range(K), counts, color="steelblue", width=1.0, linewidth=0)
    axes[0].set_title(f"Codebook Utilization: {active}/{K} = {100*util:.1f}%")
    axes[0].set_xlabel("Code Index")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    # Per-class code coverage
    unique_labels = np.unique(labels)
    coverage = []
    cls_names = []
    for lbl in unique_labels:
        mask = labels == lbl
        cls_codes = codes[mask].reshape(-1).numpy()
        cls_counts = np.bincount(cls_codes, minlength=K)
        cls_active = (cls_counts > 0).sum() / K
        coverage.append(cls_active)
        cls_names.append(MODELNET40_CLASSES[lbl])

    axes[1].barh(cls_names, coverage, color="coral")
    axes[1].set_title("Code Coverage per Class")
    axes[1].set_xlabel("Fraction of Codebook Used")
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(coverage):
        axes[1].text(v + 0.01, i, f"{100*v:.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "codebook_utilization.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Vis] Codebook utilization → {path}")
    print(f"[Vis] Overall codebook utilization: {100*util:.1f}%")
    return util


def plot_fingerprint_retrieval(fps, labels, output_dir, n_queries=5):
    """For each query, find top-5 nearest by cosine similarity."""
    fps_norm = fps / (np.linalg.norm(fps, axis=1, keepdims=True) + 1e-8)
    sim = fps_norm @ fps_norm.T  # [n, n]

    fig, axes = plt.subplots(n_queries, 6, figsize=(14, 3 * n_queries))
    rng = np.random.RandomState(0)
    query_ids = rng.choice(len(fps), n_queries, replace=False)

    for row, qid in enumerate(query_ids):
        sims = sim[qid].copy()
        sims[qid] = -1  # exclude self
        top5 = np.argsort(sims)[::-1][:5]

        q_label = MODELNET40_CLASSES[labels[qid]]
        axes[row, 0].set_facecolor("#ffe0e0")
        axes[row, 0].text(0.5, 0.5, f"Query\n{q_label}\n(#{qid})",
                          ha="center", va="center", fontsize=9, fontweight="bold",
                          transform=axes[row, 0].transAxes)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col, rid in enumerate(top5):
            r_label = MODELNET40_CLASSES[labels[rid]]
            match = labels[qid] == labels[rid]
            color = "#e0ffe0" if match else "#ffe0e0"
            axes[row, col + 1].set_facecolor(color)
            axes[row, col + 1].text(0.5, 0.5,
                                     f"{'✓' if match else '✗'} {r_label}\nsim={sims[rid]:.3f}",
                                     ha="center", va="center", fontsize=8,
                                     transform=axes[row, col + 1].transAxes)
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])

    plt.suptitle("Fingerprint Retrieval: Top-5 Nearest Neighbors", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "fingerprint_retrieval.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Vis] Fingerprint retrieval → {path}")

    # Top-5 recall
    correct = 0
    for qid in range(len(fps)):
        sims_q = sim[qid].copy()
        sims_q[qid] = -1
        top5 = np.argsort(sims_q)[::-1][:5]
        if labels[qid] in labels[top5]:
            correct += 1
    recall = correct / len(fps)
    print(f"[Vis] Top-5 retrieval recall: {100*recall:.1f}%")
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = SmallModelConfig()
    cfg.train.cache_dir = CACHE_DIR

    print("[Vis] Building model...")
    model = MaskedVQVAE3D(cfg).to(DEVICE)

    if args.checkpoint and not args.random:
        print(f"[Vis] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    print("[Vis] Loading dataset...")
    ds = ModelNet40Dataset(
        data_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
        split="val",
        use_augmentation=False,
        use_contrastive=False,
        seed=42,
    )

    print(f"[Vis] Collecting embeddings from {min(args.n_samples, len(ds))} samples...")
    fps, labels, codes = collect_embeddings(model, ds, n=args.n_samples)
    print(f"[Vis] Collected {len(fps)} embeddings, fingerprint dim={fps.shape[1]}")

    K = model.quantizer.num_embeddings
    codebook = model.quantizer.embed.cpu().numpy()

    plot_umap_fingerprints(fps, labels, OUTPUT_DIR)
    plot_umap_codebook(codebook, OUTPUT_DIR)
    util = plot_codebook_utilization(codes, K, labels, OUTPUT_DIR)
    recall = plot_fingerprint_retrieval(fps, labels, OUTPUT_DIR)

    print("\n[Vis] === Summary ===")
    print(f"  Codebook utilization: {100*util:.1f}%")
    print(f"  Top-5 retrieval recall: {100*recall:.1f}%")
    print(f"  Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
