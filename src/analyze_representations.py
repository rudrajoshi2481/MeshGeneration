"""
analyze_representations.py
---------------------------
Loads the best MeshGPT checkpoint and evaluates three representations:
  1. code_indices   — discrete codebook tokens [B, 4096]  → histogram [B, 256]
  2. fingerprint    — attention-weighted geometry vector   [B, 128]
  3. quant_pooled   — mean-pooled quantized features      [B, 64]

Metrics per representation:
  - t-SNE plot (coloured by class)
  - Silhouette score
  - Top-5 FAISS retrieval accuracy
  - Code consistency score (codebook only)

All outputs go to:  <run_dir>/trash/representation_analysis/
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch.serialization

# ── insert src on path ──────────────────────────────────────────────────────
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from config import SmallModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset
from preprocessing import MODELNET40_CLASSES

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not found — skipping silhouette + t-SNE")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[WARN] faiss not found — using torch brute-force for retrieval")

# ── pretty class names ────────────────────────────────────────────────────────
CLASS_NAMES = MODELNET40_CLASSES  # list of 40 strings


# ────────────────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_representations(model, loader, device, num_classes=40, codebook_size=256):
    """
    Runs the whole val set through the model and collects:
      code_hist   : [N, codebook_size]  weighted histogram of codebook usage
      fingerprint : [N, fp_dim]
      quant_pooled: [N, emb_dim]        mean-pooled quantized vectors
      labels      : [N]
    """
    model.eval()
    model = model.to(device)

    code_hists, fingerprints, quant_pools, labels_all = [], [], [], []

    for batch in loader:
        pts  = batch["points"].to(device)
        nrm  = batch["normals"].to(device)
        curv = batch["curvature"].to(device)
        qtg  = batch["query_pts"].to(device)
        lbl  = batch["label"].to(device)

        enc = model.encode_and_quantize(pts, nrm, curv, lbl)

        # ── 1. Codebook index histogram ─────────────────────────────────────
        code_idx = enc["code_all"]          # [B, N_vox]  long
        B = code_idx.shape[0]
        # Weighted by masker score so geometry-important regions count more
        scores = enc["masker_out"]["score_map"]  # [B, N_vox]
        hist = torch.zeros(B, codebook_size, device=device)
        for b in range(B):
            hist[b].scatter_add_(0, code_idx[b], scores[b])
        # L1-normalise
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        code_hists.append(hist.cpu())

        # ── 2. Fingerprint ──────────────────────────────────────────────────
        quant_kept = enc["quant_kept"]      # [B, K, d_q]
        fp_vec, _ = model.fingerprint(quant_kept)   # [B, fp_dim]
        fingerprints.append(fp_vec.cpu())

        # ── 3. Mean-pooled quantized features ──────────────────────────────
        quant_all = enc["quant_all"]        # [B, N_vox, d_q]
        qp = quant_all.mean(dim=1)          # [B, d_q]
        quant_pools.append(qp.cpu())

        labels_all.append(lbl.cpu())

    return (
        torch.cat(code_hists,   dim=0).numpy().astype("float32"),
        torch.cat(fingerprints, dim=0).numpy().astype("float32"),
        torch.cat(quant_pools,  dim=0).numpy().astype("float32"),
        torch.cat(labels_all,   dim=0).numpy(),
    )


# ────────────────────────────────────────────────────────────────────────────
# Retrieval (FAISS or torch)
# ────────────────────────────────────────────────────────────────────────────

def top_k_retrieval(reps: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """
    For each query, retrieve top-k nearest neighbours (excluding self).
    Returns fraction of queries where ≥1 retrieved neighbour shares the class.
    """
    N, D = reps.shape
    if HAS_FAISS:
        index = faiss.IndexFlatL2(D)
        index.add(reps)
        _, indices = index.search(reps, k + 1)   # +1 to exclude self
        indices = indices[:, 1:]                  # drop self (rank-0)
    else:
        # Brute-force cosine similarity via torch
        T = torch.from_numpy(reps)
        T = F.normalize(T, dim=1)
        sim = torch.mm(T, T.T)                    # [N, N]
        sim.fill_diagonal_(-1e9)
        _, indices = sim.topk(k, dim=1)
        indices = indices.numpy()

    correct = 0
    for i in range(N):
        retrieved = labels[indices[i]]
        if labels[i] in retrieved:
            correct += 1
    return correct / N


# ────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────────────────────

def plot_tsne(reps: np.ndarray, labels: np.ndarray, title: str,
              out_path: str, n_classes: int = 40):
    if not HAS_SKLEARN:
        return
    print(f"  [t-SNE] fitting {title} ...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1500,
                random_state=42, n_jobs=-1)
    emb = tsne.fit_transform(reps)

    cmap = cm.get_cmap("tab20", n_classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(emb[:, 0], emb[:, 1],
                         c=labels, cmap=cmap,
                         s=12, alpha=0.7, linewidths=0)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(0, n_classes, 2))
    cbar.set_label("Class ID")
    ax.set_title(f"t-SNE — {title}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [t-SNE] saved → {out_path}")


def plot_per_class_retrieval(per_class_acc: dict, title: str, out_path: str):
    classes = sorted(per_class_acc.keys())
    accs = [per_class_acc[c] for c in classes]
    names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(range(len(classes)), accs, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-5 Retrieval Accuracy")
    ax.set_title(f"Per-class Top-5 Retrieval — {title}")
    ax.axhline(y=1/40, color="red", linestyle="--", linewidth=1, label="Random")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [per-class retrieval] saved → {out_path}")


def plot_code_consistency(consistency: dict, out_path: str):
    classes = sorted(consistency.keys())
    scores = [consistency[c] for c in classes]
    names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(classes)), scores, color="coral", alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Code Consistency Score")
    ax.set_title("Codebook — Per-class Code Consistency (top-20 codes overlap)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [code consistency] saved → {out_path}")


def plot_summary_bar(results: dict, out_path: str):
    """Single bar chart comparing silhouette + retrieval across representations."""
    reps = ["code_hist", "fingerprint", "quant_pooled"]
    metrics = ["silhouette", "retrieval_top5"]
    labels_bar = [f"{r}\n{m}" for r in reps for m in metrics]
    values = []
    for r in reps:
        for m in metrics:
            key = f"{r}_{m}"
            values.append(results.get(key, 0.0))

    colours = ["#4C72B0", "#DD8452"] * len(reps)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(values)), values, color=colours, alpha=0.85)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels_bar, fontsize=9)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Representation Quality Summary")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [summary] saved → {out_path}")


def plot_codebook_class_heatmap(code_hists: np.ndarray, labels: np.ndarray,
                                out_path: str, top_k: int = 50):
    """
    Heatmap: rows = 40 classes, columns = top-50 most-used codes.
    Shows which codes are class-specific vs shared.
    """
    n_codes = code_hists.shape[1]
    # Average histogram per class
    class_hists = np.zeros((40, n_codes))
    for c in range(40):
        mask = labels == c
        if mask.sum() > 0:
            class_hists[c] = code_hists[mask].mean(0)
    # Select top-k most variable codes
    code_var = class_hists.var(axis=0)
    top_codes = np.argsort(code_var)[::-1][:top_k]
    sub = class_hists[:, top_codes]

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(sub, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Avg weighted usage")
    ax.set_yticks(range(40))
    ax.set_yticklabels([CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
                        for c in range(40)], fontsize=7)
    ax.set_xlabel(f"Top-{top_k} most class-discriminative codes")
    ax.set_title("Codebook Usage Heatmap (class × code)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [heatmap] saved → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Code consistency metric
# ────────────────────────────────────────────────────────────────────────────

def compute_code_consistency(code_hists: np.ndarray, labels: np.ndarray,
                             top_k: int = 20) -> dict:
    """
    For each class, find the top-k codes by mean usage.
    Consistency = fraction of top-k class codes that each sample also has in its top-k.
    Returns per-class mean consistency.
    """
    n_classes = int(labels.max()) + 1
    consistency = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() < 2:
            continue
        class_hist = code_hists[mask]
        mean_hist = class_hist.mean(0)
        top_codes = set(np.argsort(mean_hist)[::-1][:top_k])
        per_sample = []
        for h in class_hist:
            sample_top = set(np.argsort(h)[::-1][:top_k])
            overlap = len(top_codes & sample_top) / top_k
            per_sample.append(overlap)
        consistency[c] = float(np.mean(per_sample))
    return consistency


def per_class_retrieval(reps: np.ndarray, labels: np.ndarray, k: int = 5) -> dict:
    """Top-5 retrieval accuracy per class."""
    N, D = reps.shape
    if HAS_FAISS:
        index = faiss.IndexFlatL2(D)
        index.add(reps)
        _, indices = index.search(reps, k + 1)
        indices = indices[:, 1:]
    else:
        T = F.normalize(torch.from_numpy(reps), dim=1)
        sim = torch.mm(T, T.T)
        sim.fill_diagonal_(-1e9)
        _, indices = sim.topk(k, dim=1)
        indices = indices.numpy()

    per_class = defaultdict(list)
    for i in range(N):
        retrieved = labels[indices[i]]
        hit = int(labels[i] in retrieved)
        per_class[int(labels[i])].append(hit)
    return {c: float(np.mean(v)) for c, v in per_class.items()}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
        default="/data/joshi/MESHGPT/new_implementation/runs/run_20260310_040537/"
                "checkpoints/small-epoch=0070-val/iou=0.4227-val/cls_acc=0.7000.ckpt",
        help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str,
        default="/data/joshi/modelnet40_meshes")
    parser.add_argument("--cache_dir", type=str,
        default="/data/joshi/MESHGPT/new_implementation/trash/cache")
    parser.add_argument("--out_dir", type=str,
        default="/data/joshi/MESHGPT/new_implementation/trash/representation_analysis")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", type=str, default="val",
        help="Which split to evaluate on: val or train")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    cfg = SmallModelConfig()
    # PyTorch 2.6 changed weights_only default to True; monkey-patch to False
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    model = MaskedVQVAE3D.load_from_checkpoint(
        args.ckpt, cfg=cfg, strict=False,
        map_location=device,
    )
    torch.load = _orig_load  # restore
    model.eval()
    model = model.to(device)
    print("[INFO] Model loaded successfully")

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"[INFO] Loading {args.split} dataset from {args.data_dir}")
    dataset = ModelNet40Dataset(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        use_augmentation=False,
        use_contrastive=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[INFO] {len(dataset)} samples, {len(loader)} batches")

    # ── Extract ───────────────────────────────────────────────────────────
    print("[INFO] Extracting representations ...")
    code_hists, fingerprints, quant_pools, labels = extract_representations(
        model, loader, device,
        codebook_size=cfg.vq.num_embeddings
    )
    print(f"[INFO] code_hist   shape: {code_hists.shape}")
    print(f"[INFO] fingerprint shape: {fingerprints.shape}")
    print(f"[INFO] quant_pool  shape: {quant_pools.shape}")
    print(f"[INFO] labels      shape: {labels.shape}")

    results = {}

    # ── Per-representation analysis ───────────────────────────────────────
    reps_dict = {
        "code_hist":   code_hists,
        "fingerprint": fingerprints,
        "quant_pooled": quant_pools,
    }

    for name, reps in reps_dict.items():
        print(f"\n{'='*60}")
        print(f"  Analysing: {name}  shape={reps.shape}")
        print(f"{'='*60}")

        # 1. Silhouette score
        if HAS_SKLEARN:
            sil = silhouette_score(reps, labels, sample_size=min(2000, len(labels)),
                                   random_state=42)
            results[f"{name}_silhouette"] = float(sil)
            print(f"  Silhouette score : {sil:.4f}  (higher=better, max=1.0)")

        # 2. Top-5 FAISS retrieval
        ret5 = top_k_retrieval(reps, labels, k=5)
        results[f"{name}_retrieval_top5"] = float(ret5)
        print(f"  Top-5 retrieval  : {ret5:.4f}  (random={1/40:.4f})")

        # 3. Top-10 retrieval for extra granularity
        ret10 = top_k_retrieval(reps, labels, k=10)
        results[f"{name}_retrieval_top10"] = float(ret10)
        print(f"  Top-10 retrieval : {ret10:.4f}")

        # 4. t-SNE
        plot_tsne(reps, labels, name,
                  os.path.join(args.out_dir, f"tsne_{name}.png"))

        # 5. Per-class retrieval
        pc_ret = per_class_retrieval(reps, labels, k=5)
        results[f"{name}_per_class_retrieval"] = pc_ret
        plot_per_class_retrieval(pc_ret, name,
            os.path.join(args.out_dir, f"per_class_retrieval_{name}.png"))

    # ── Codebook-specific metrics ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Codebook-specific analysis")
    print(f"{'='*60}")
    consistency = compute_code_consistency(code_hists, labels, top_k=20)
    avg_consistency = float(np.mean(list(consistency.values())))
    results["code_hist_consistency"] = avg_consistency
    print(f"  Avg code consistency : {avg_consistency:.4f}")

    plot_code_consistency(consistency,
        os.path.join(args.out_dir, "code_consistency_per_class.png"))

    plot_codebook_class_heatmap(code_hists, labels,
        os.path.join(args.out_dir, "codebook_heatmap.png"), top_k=50)

    # ── Summary plot ──────────────────────────────────────────────────────
    plot_summary_bar(results, os.path.join(args.out_dir, "summary_comparison.png"))

    # ── Print final ranking ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL RANKING (Top-5 Retrieval Accuracy)")
    print(f"{'='*60}")
    ranking = sorted(
        [(k, v) for k, v in results.items() if "retrieval_top5" in k],
        key=lambda x: x[1], reverse=True
    )
    for i, (k, v) in enumerate(ranking):
        print(f"  #{i+1}  {k:<40s}  {v:.4f}")

    if HAS_SKLEARN:
        print(f"\n  FINAL RANKING (Silhouette Score)")
        sil_ranking = sorted(
            [(k, v) for k, v in results.items() if "silhouette" in k],
            key=lambda x: x[1], reverse=True
        )
        for i, (k, v) in enumerate(sil_ranking):
            print(f"  #{i+1}  {k:<40s}  {v:.4f}")

    # ── SEDD compatibility note ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SEDD INTEGRATION NOTES")
    print(f"{'='*60}")
    best_ret = ranking[0][0].replace("_retrieval_top5", "")
    print(f"  Best for retrieval/semantic  : {best_ret}")
    print(f"  Codebook vocab size          : {cfg.vq.num_embeddings}")
    print(f"  Sequence length (grid tokens): {cfg.encoder.grid_res**3}")
    print(f"  SEDD input: code_indices [B, {cfg.encoder.grid_res**3}] (long, 0-{cfg.vq.num_embeddings-1})")
    print(f"  SEDD vocab_size = {cfg.vq.num_embeddings}, mask_id = {cfg.vq.num_embeddings}")
    print(f"  First N tokens → give to SEDD, mask rest → model fills them in")

    # ── Save JSON report ───────────────────────────────────────────────────
    # Remove non-serialisable per-class dicts from top-level
    report = {k: v for k, v in results.items()
              if not k.endswith("_per_class_retrieval")}
    report["checkpoint"] = args.ckpt
    report["split"] = args.split
    report["n_samples"] = int(len(labels))

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved → {report_path}")
    print(f"[INFO] All plots saved → {args.out_dir}/")
    print("[DONE]")


if __name__ == "__main__":
    main()
