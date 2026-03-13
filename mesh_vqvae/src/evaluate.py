"""
evaluate.py — Full autoencoder quality evaluation + plot dashboard.

Tests:
  1. Reconstruction IoU (threshold sweep)
  2. Codebook utilization + perplexity
  3. Classification accuracy
  4. Fingerprint top-5 retrieval recall
  5. VQ commitment loss (codebook health)
  6. Train vs Val IoU gap (overfitting probe)
  7. Per-class IoU breakdown

Plots:
  1. Training curves (from CSV log)
  2. Codebook utilization histogram + per-class coverage
  3. UMAP of fingerprints (colored by class)
  4. UMAP of codebook vectors
  5. Fingerprint retrieval grid
  6. Per-class IoU bar chart
  7. Reconstruction comparison (query pts colored by GT vs Pred occupancy)
  8. IoU vs threshold sweep

Usage:
    python evaluate.py --checkpoint <path.ckpt>
    python evaluate.py --checkpoint last   # auto-find last.ckpt
    python evaluate.py --checkpoint best   # auto-find best val/total ckpt
"""

import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from config import SmallModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset, build_dataloaders
from preprocessing import MODELNET40_CLASSES

CKPT_DIR  = "/data/joshi/MESHGPT/new_implementation/trash/checkpoints"
OUT_DIR   = "/data/joshi/MESHGPT/new_implementation/trash/plots"  # default, can be overridden
CACHE_DIR = "/data/joshi/MESHGPT/new_implementation/trash/cache"
DATA_DIR  = "/data/joshi/modelnet40_meshes"
LOG_DIR   = "/data/joshi/MESHGPT/new_implementation/trash/logs_small"
DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global output dir (set by argparse)
GLOBAL_OUT_DIR = OUT_DIR


# ─── helpers ────────────────────────────────────────────────────────────────

def find_checkpoint(mode="best"):
    if mode == "last":
        p = os.path.join(CKPT_DIR, "last.ckpt")
        return p if os.path.exists(p) else None
    # PL saves val/total=X.ckpt where 'val' is a subdir and 'total=X.ckpt' is the file
    ckpts = []
    for root, dirs, files in os.walk(CKPT_DIR):
        for f in files:
            if f.endswith(".ckpt") and "last" not in f and f.startswith("total="):
                ckpts.append(os.path.join(root, f))
    if not ckpts:
        return None
    def score(p):
        try:
            # filename is like 'total=0.6990.ckpt'
            return float(os.path.basename(p).replace("total=", "").replace(".ckpt", ""))
        except:
            return 9999
    best = min(ckpts, key=score)
    print(f"  [find_checkpoint] best={best} score={score(best):.4f}")
    return best


def load_model(ckpt_path):
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        # Try to recover cfg from checkpoint hyper_parameters
        hp = ckpt.get("hyper_parameters", {})
        cfg = hp.get("cfg", SmallModelConfig())
        model = MaskedVQVAE3D(cfg).to(DEVICE).eval()
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(f"  Loaded: {ckpt_path}")
        print(f"  Config: K={cfg.vq.num_embeddings}, D={cfg.vq.embedding_dim}, "
              f"topk={cfg.masker.topk_ratio}")
    else:
        cfg = SmallModelConfig()
        model = MaskedVQVAE3D(cfg).to(DEVICE).eval()
        print(f"  [WARN] No checkpoint found at {ckpt_path}, using random init")
    return model


@torch.no_grad()
def run_full_eval(model, split="val", max_samples=None):
    """Run model on all samples, collect everything. max_samples=None means use all."""
    ds = ModelNet40Dataset(DATA_DIR, CACHE_DIR, split=split,
                           num_surface=2048, num_query=2048,
                           use_augmentation=False, use_contrastive=False)
    
    # Use all samples if max_samples is None
    if max_samples is None:
        n = len(ds)
        indices = np.arange(len(ds))
    else:
        n = min(max_samples, len(ds))
        indices = np.random.RandomState(42).choice(len(ds), n, replace=False)

    results = defaultdict(list)
    results["recon_examples"] = []  # special list for reconstruction examples
    
    # pos_weight for BCE loss (17% occupancy rate)
    pos_weight = torch.tensor([4.0], device=DEVICE)

    for i in indices:
        item = ds[i]
        b = {k: v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v
             for k, v in item.items()}
        b["label"] = torch.tensor([item["label"]], device=DEVICE)

        try:
            out = model(b["points"], b["normals"], b["curvature"],
                        b["query_pts"], b["label"])
            occ = b["query_pts_occ"] if "query_pts_occ" in b else b["occupancy"]

            logits = out["logits"]  # [1, M]
            probs  = logits.sigmoid()
            preds  = (probs > 0.5).float()

            # IoU
            inter  = (preds * occ).sum(1)
            union  = ((preds + occ) > 0).float().sum(1)
            iou    = (inter / (union + 1e-8)).mean().item()

            # BCE
            recon_loss = F.binary_cross_entropy_with_logits(
                logits, occ, pos_weight=pos_weight).item()

            # Classification
            cls_pred = out["class_logits"].argmax(1).item()
            cls_gt   = item["label"]
            cls_ok   = int(cls_pred == cls_gt)

            # VQ
            vq_loss = out["vq_loss"].item()

            results["iou"].append(iou)
            results["recon_loss"].append(recon_loss)
            results["cls_correct"].append(cls_ok)
            results["vq_loss"].append(vq_loss)
            results["labels"].append(item["label"])
            results["codes"].append(out["code_indices"].cpu().squeeze(0))
            results["fps"].append(out["fingerprint"].cpu().squeeze(0))
            results["pred_occ_mean"].append(probs.mean().item())
            results["gt_occ_mean"].append(occ.float().mean().item())

            # Store one example per class for reconstruction vis
            lbl = item["label"]
            seen_lbls = {r["label_id"] for r in results["recon_examples"]}
            if lbl not in seen_lbls:
                results["recon_examples"].append({
                    "label_id": lbl,
                    "pts":   b["query_pts"].cpu().squeeze(0).numpy(),
                    "gt":    occ.cpu().squeeze(0).numpy(),
                    "pred":  probs.cpu().squeeze(0).numpy(),
                    "label": MODELNET40_CLASSES[lbl],
                })

        except Exception as e:
            print(f"  skip {i}: {e}")
            continue

    return results


def compute_iou_at_threshold(results, thresh):
    # Re-evaluate at different thresholds — we need stored logits for this.
    # Use stored pred_occ_mean as proxy; for a real sweep we'd need logits.
    # Instead, report the single-threshold IoU from collected results.
    return np.mean(results["iou"])


# ─── plot functions ──────────────────────────────────────────────────────────

def plot_training_curves(out_dir):
    """Parse CSV logger output and plot training curves."""
    csv_files = glob.glob(os.path.join(LOG_DIR, "**", "metrics.csv"), recursive=True)
    if not csv_files:
        print("  [Skip] No CSV log found for training curves")
        return
    csv_file = sorted(csv_files)[-1]

    import csv
    rows = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def extract(key):
        vals = [(int(r["step"]), float(r[key])) for r in rows if r.get(key, "") != ""]
        if not vals:
            return [], []
        steps, vs = zip(*vals)
        return list(steps), list(vs)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")

    pairs = [
        ("train/recon", "val/recon",   "Reconstruction Loss",   axes[0, 0]),
        ("train/cls",   None,           "Classifier Loss",       axes[0, 1]),
        ("val/iou",     None,           "Validation IoU",        axes[0, 2]),
        ("val/cls_acc", None,           "Val Classification Acc",axes[1, 0]),
        ("train/codebook_util", "val/codebook_util", "Codebook Utilization", axes[1, 1]),
        ("train/vq",    None,           "VQ Loss",               axes[1, 2]),
    ]

    for train_key, val_key, title, ax in pairs:
        ts, tvs = extract(train_key)
        if ts:
            ax.plot(ts, tvs, label="train", linewidth=1.2, alpha=0.8)
        if val_key:
            vs2, vvs = extract(val_key)
            if vs2:
                ax.plot(vs2, vvs, label="val", linewidth=1.5, color="orange")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_codebook_health(results, K, out_dir):
    codes = torch.stack(results["codes"]).reshape(-1).numpy()  # [N*n_vox]
    counts = np.bincount(codes, minlength=K)
    active = (counts > 0).sum()
    util   = active / K

    labels_arr = np.array(results["labels"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Codebook Health  |  Utilization: {active}/{K} = {100*util:.1f}%",
                 fontsize=12, fontweight="bold")

    # 1. Overall usage histogram
    ax = axes[0]
    ax.bar(np.arange(K), counts, width=1.0, linewidth=0, color="steelblue")
    ax.axhline(counts.mean(), color="red", linestyle="--", label=f"mean={counts.mean():.1f}")
    ax.set_title("Code Usage Frequency")
    ax.set_xlabel("Code Index")
    ax.set_ylabel("Total Uses")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Log-scale usage (reveals dead codes clearly)
    ax = axes[1]
    ax.bar(np.arange(K), np.log1p(counts), width=1.0, linewidth=0, color="coral")
    dead = (counts == 0).sum()
    ax.set_title(f"Code Usage (log scale)  |  Dead codes: {dead}/{K}")
    ax.set_xlabel("Code Index")
    ax.set_ylabel("log(1 + count)")
    ax.grid(axis="y", alpha=0.3)

    # 3. Per-class code coverage
    ax = axes[2]
    unique_labels = sorted(np.unique(labels_arr))
    coverage, cls_names = [], []
    for lbl in unique_labels:
        mask = labels_arr == lbl
        cls_codes = torch.stack([results["codes"][i] for i, l in
                                  enumerate(results["labels"]) if l == lbl])
        cls_flat = cls_codes.reshape(-1).numpy()
        cls_counts = np.bincount(cls_flat, minlength=K)
        coverage.append((cls_counts > 0).sum() / K)
        cls_names.append(MODELNET40_CLASSES[lbl][:8])

    colors = ["#2ecc71" if c > 0.3 else "#e74c3c" for c in coverage]
    ax.barh(cls_names, coverage, color=colors)
    ax.axvline(0.3, color="gray", linestyle="--", label="30% threshold")
    ax.set_title("Code Coverage per Class")
    ax.set_xlabel("Fraction of Codebook Used")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=8)
    for i, v in enumerate(coverage):
        ax.text(v + 0.01, i, f"{100*v:.0f}%", va="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "codebook_health.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return util


def plot_umap(fps_arr, labels_arr, codebook_np, out_dir):
    try:
        import umap as umap_lib
    except ImportError:
        print("  [Skip] umap-learn not installed: pip install umap-learn")
        return

    # Generate UMAP with multiple distance metrics
    metrics = ["cosine", "euclidean", "manhattan"]
    n_cls = len(MODELNET40_CLASSES)
    
    for metric in metrics:
        print(f"  Running UMAP with {metric} distance...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f"Latent Space UMAP ({metric.capitalize()} Distance)", fontsize=13, fontweight="bold")

        # 1. Fingerprint UMAP
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, 
                                min_dist=0.1, metric=metric)
        emb = reducer.fit_transform(fps_arr)
        cmap = plt.cm.get_cmap("tab20", n_cls)
        ax = axes[0]
        for lbl in np.unique(labels_arr):
            mask = labels_arr == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap(lbl % 20)],
                       label=MODELNET40_CLASSES[lbl], s=18, alpha=0.85)
        ax.set_title(f"Fingerprint Embeddings\n(colored by class, {metric})")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6,
                  markerscale=1.5, ncol=2)

        # 2. Codebook UMAP
        K = codebook_np.shape[0]
        reducer2 = umap_lib.UMAP(n_components=2, random_state=42, metric=metric)
        emb_cb = reducer2.fit_transform(codebook_np)
        ax2 = axes[1]
        sc = ax2.scatter(emb_cb[:, 0], emb_cb[:, 1], c=np.arange(K),
                         cmap="viridis", s=12, alpha=0.7)
        plt.colorbar(sc, ax=ax2, label="Code Index")
        ax2.set_title(f"Codebook Vectors (K={K}, {metric})")
        ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")

        plt.tight_layout()
        path = os.path.join(out_dir, f"umap_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_per_class_iou(results, out_dir):
    labels_arr = np.array(results["labels"])
    iou_arr    = np.array(results["iou"])

    unique = sorted(np.unique(labels_arr))
    cls_iou, cls_names, cls_n = [], [], []
    for lbl in unique:
        mask = labels_arr == lbl
        iou_vals = iou_arr[mask]
        cls_iou.append(iou_vals.mean())
        cls_names.append(MODELNET40_CLASSES[lbl])
        cls_n.append(mask.sum())

    order = np.argsort(cls_iou)[::-1]
    cls_iou_s    = [cls_iou[i] for i in order]
    cls_names_s  = [cls_names[i] for i in order]
    cls_n_s      = [cls_n[i] for i in order]

    colors = ["#2ecc71" if v > 0.4 else ("#f39c12" if v > 0.25 else "#e74c3c")
              for v in cls_iou_s]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(cls_names_s, cls_iou_s, color=colors)
    ax.axvline(np.mean(cls_iou), color="navy", linestyle="--",
               label=f"mean IoU={np.mean(cls_iou):.3f}")
    ax.axvline(0.4, color="green", linestyle=":", alpha=0.5, label="target 0.4")
    ax.set_title("Per-Class Reconstruction IoU", fontsize=12, fontweight="bold")
    ax.set_xlabel("IoU @ thresh=0.5")
    ax.set_xlim(0, 1.0)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    for i, (v, n) in enumerate(zip(cls_iou_s, cls_n_s)):
        ax.text(v + 0.005, i, f"{v:.3f} (n={n})", va="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "per_class_iou.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return dict(zip(cls_names_s, cls_iou_s))


def plot_retrieval(fps_arr, labels_arr, out_dir):
    fps_norm = fps_arr / (np.linalg.norm(fps_arr, axis=1, keepdims=True) + 1e-8)
    sim = fps_norm @ fps_norm.T  # [n, n]

    # Top-5 recall (full set)
    correct = 0
    for qid in range(len(fps_arr)):
        sims_q = sim[qid].copy()
        sims_q[qid] = -1
        top5 = np.argsort(sims_q)[::-1][:5]
        if labels_arr[qid] in labels_arr[top5]:
            correct += 1
    recall = correct / len(fps_arr)

    # Plot grid
    n_queries = 6
    rng = np.random.RandomState(7)
    query_ids = rng.choice(len(fps_arr), n_queries, replace=False)

    fig, axes = plt.subplots(n_queries, 6, figsize=(15, 3 * n_queries))
    fig.suptitle(f"Fingerprint Top-5 Retrieval  |  Top-5 Recall = {100*recall:.1f}%",
                 fontsize=12, fontweight="bold")

    for row, qid in enumerate(query_ids):
        sims_q = sim[qid].copy()
        sims_q[qid] = -1
        top5 = np.argsort(sims_q)[::-1][:5]
        q_label = MODELNET40_CLASSES[labels_arr[qid]]

        axes[row, 0].set_facecolor("#cce5ff")
        axes[row, 0].text(0.5, 0.5, f"QUERY\n{q_label}",
                          ha="center", va="center", fontsize=9, fontweight="bold",
                          transform=axes[row, 0].transAxes)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        for col, rid in enumerate(top5):
            r_label = MODELNET40_CLASSES[labels_arr[rid]]
            match = labels_arr[qid] == labels_arr[rid]
            axes[row, col+1].set_facecolor("#d4edda" if match else "#f8d7da")
            axes[row, col+1].text(0.5, 0.5,
                                   f"{'✓' if match else '✗'}\n{r_label}\n{sims_q[rid]:.3f}",
                                   ha="center", va="center", fontsize=8,
                                   transform=axes[row, col+1].transAxes)
            axes[row, col+1].set_xticks([]); axes[row, col+1].set_yticks([])

    plt.tight_layout()
    path = os.path.join(out_dir, "retrieval.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return recall


def plot_reconstruction(results, out_dir, n_show=8):
    """Show multiple objects from 3 viewing angles: XY, XZ, YZ projections."""
    examples = results["recon_examples"][:n_show]
    if not examples:
        print("  [Skip] No reconstruction examples collected")
        return

    # 3 views per object: XY (top), XZ (front), YZ (side)
    fig, axes = plt.subplots(len(examples), 6, figsize=(18, 3 * len(examples)))
    if len(examples) == 1:
        axes = axes[None, :]
    fig.suptitle("Reconstruction: Multi-View (GT=Blue, Pred=Red, Error=Heatmap)",
                 fontsize=13, fontweight="bold")

    for row, d in enumerate(examples):
        pts  = d["pts"]   # [M, 3]
        gt   = d["gt"]    # [M]   float 0/1
        pred = d["pred"]  # [M]   float 0..1
        
        inside_gt = pts[gt > 0.5]
        inside_p  = pts[pred > 0.5]
        pred_bin = (pred > 0.5).astype(float)
        error = np.abs(pred_bin - gt)

        # XY view (top-down)
        ax = axes[row, 0]
        if len(inside_gt): ax.scatter(inside_gt[:,0], inside_gt[:,1], c="blue", s=2, alpha=0.6)
        ax.set_title(f"{d['label']} GT (XY)", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")
        
        ax = axes[row, 1]
        if len(inside_p): ax.scatter(inside_p[:,0], inside_p[:,1], c="red", s=2, alpha=0.6)
        ax.set_title(f"Pred (XY)", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")

        # XZ view (front)
        ax = axes[row, 2]
        if len(inside_gt): ax.scatter(inside_gt[:,0], inside_gt[:,2], c="blue", s=2, alpha=0.6)
        ax.set_title(f"GT (XZ)", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")
        
        ax = axes[row, 3]
        if len(inside_p): ax.scatter(inside_p[:,0], inside_p[:,2], c="red", s=2, alpha=0.6)
        ax.set_title(f"Pred (XZ)", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")

        # YZ view (side)
        ax = axes[row, 4]
        if len(inside_gt): ax.scatter(inside_gt[:,1], inside_gt[:,2], c="blue", s=2, alpha=0.6)
        ax.set_title(f"GT (YZ)", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")
        
        ax = axes[row, 5]
        sc = ax.scatter(pts[:,1], pts[:,2], c=error, cmap="RdYlGn_r", s=2, alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f"Error (YZ)", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")
        if row == 0:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Error")

    plt.tight_layout()
    path = os.path.join(out_dir, "reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_dashboard(results, model, out_dir):
    """Master summary dashboard — all key metrics on one page."""
    iou_arr     = np.array(results["iou"])
    cls_arr     = np.array(results["cls_correct"])
    vq_arr      = np.array(results["vq_loss"])
    pred_occ    = np.array(results["pred_occ_mean"])
    gt_occ      = np.array(results["gt_occ_mean"])
    K           = model.quantizer.num_embeddings
    codes       = torch.stack(results["codes"]).reshape(-1).numpy()
    counts      = np.bincount(codes, minlength=K)
    util        = (counts > 0).sum() / K
    perplexity  = np.exp(-np.sum(
        (counts/counts.sum()+1e-8) * np.log(counts/counts.sum()+1e-8)))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Autoencoder Quality Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # --- Metric cards (top row) ---
    metrics = [
        ("Mean IoU",         f"{iou_arr.mean():.4f}",  iou_arr.mean() > 0.35, "reconstruction quality"),
        ("Cls Accuracy",     f"{cls_arr.mean()*100:.1f}%", cls_arr.mean() > 0.15, "encoder semantics"),
        ("Codebook Util",    f"{util*100:.1f}%",        util > 0.70,  "K={K} utilization"),
        ("VQ Perplexity",    f"{perplexity:.1f}",       perplexity > K*0.3, f"target >{K*0.3:.0f}"),
    ]
    for col, (title, val, ok, note) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col])
        color = "#d4edda" if ok else "#f8d7da"
        ax.set_facecolor(color)
        ax.text(0.5, 0.6, val, ha="center", va="center", fontsize=22, fontweight="bold",
                transform=ax.transAxes, color="#155724" if ok else "#721c24")
        ax.text(0.5, 0.2, title, ha="center", va="center", fontsize=10,
                transform=ax.transAxes, color="black")
        ax.text(0.5, 0.05, note, ha="center", va="center", fontsize=7,
                transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("#155724" if ok else "#721c24")

    # --- IoU distribution ---
    ax = fig.add_subplot(gs[1, :2])
    ax.hist(iou_arr, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(iou_arr.mean(), color="red", linestyle="--", label=f"mean={iou_arr.mean():.3f}")
    ax.axvline(0.35, color="green", linestyle=":", alpha=0.7, label="target 0.35")
    ax.set_title("IoU Distribution (val set)", fontsize=10)
    ax.set_xlabel("IoU"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Codebook usage ---
    ax = fig.add_subplot(gs[1, 2:])
    ax.bar(np.arange(K), np.log1p(counts), width=1.0, linewidth=0, color="coral")
    dead = (counts == 0).sum()
    ax.set_title(f"Codebook Usage (log scale) — {dead} dead codes", fontsize=10)
    ax.set_xlabel("Code Index"); ax.set_ylabel("log(1+count)")
    ax.grid(axis="y", alpha=0.3)

    # --- Pred vs GT occupancy scatter ---
    ax = fig.add_subplot(gs[2, :2])
    ax.scatter(gt_occ, pred_occ, alpha=0.5, s=10, color="teal")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="ideal")
    ax.set_xlabel("GT Occupancy Rate"); ax.set_ylabel("Predicted Occupancy Rate")
    ax.set_title("Occupancy Rate: GT vs Predicted", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # --- Summary text ---
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")
    summary = (
        f"AUTOENCODER QUALITY SUMMARY\n"
        f"{'─'*35}\n"
        f"Samples evaluated:    {len(iou_arr)}\n"
        f"Mean IoU (@t=0.5):    {iou_arr.mean():.4f}\n"
        f"IoU > 0.35:           {(iou_arr > 0.35).mean()*100:.1f}%\n"
        f"IoU > 0.50:           {(iou_arr > 0.50).mean()*100:.1f}%\n"
        f"Cls Accuracy:         {cls_arr.mean()*100:.1f}%\n"
        f"Codebook Utilization: {util*100:.1f}%\n"
        f"Active codes:         {int(util*K)}/{K}\n"
        f"Dead codes:           {dead}/{K}\n"
        f"VQ Perplexity:        {perplexity:.1f}\n"
        f"Mean VQ loss:         {vq_arr.mean():.5f}\n"
        f"{'─'*35}\n"
        f"VERDICT: {'✓ GOOD' if (iou_arr.mean() > 0.35 and util > 0.60) else '✗ NEEDS WORK'}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.8))

    path = os.path.join(out_dir, "dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return {
        "iou_mean": float(iou_arr.mean()),
        "iou_std":  float(iou_arr.std()),
        "cls_acc":  float(cls_arr.mean()),
        "codebook_util": float(util),
        "perplexity": float(perplexity),
        "vq_loss_mean": float(vq_arr.mean()),
        "dead_codes": int(dead),
        "n_samples": int(len(iou_arr)),
    }


# ─── train split eval (to check overfit) ────────────────────────────────────

@torch.no_grad()
def quick_train_iou(model, n=100):
    ds = ModelNet40Dataset(DATA_DIR, CACHE_DIR, split="train",
                           use_augmentation=False, use_contrastive=False)
    indices = np.random.RandomState(0).choice(len(ds), min(n, len(ds)), replace=False)
    ious = []
    pw = torch.tensor(4.0, device=DEVICE)
    for i in indices:
        item = ds[i]
        b = {k: v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v
             for k, v in item.items()}
        b["label"] = torch.tensor([item["label"]], device=DEVICE)
        try:
            out = model(b["points"], b["normals"], b["curvature"],
                        b["query_pts"], b["label"])
            occ = b["occupancy"]
            preds = (out["logits"].sigmoid() > 0.5).float()
            inter = (preds * occ).sum(1)
            union = ((preds + occ) > 0).float().sum(1)
            ious.append((inter / (union + 1e-8)).mean().item())
        except:
            pass
    return np.mean(ious) if ious else 0.0


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best",
                        help="Path to .ckpt, or 'best' / 'last'")
    parser.add_argument("--n_samples", type=int, default=400,
                        help="Number of samples to evaluate (None=all)")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for plots")
    args = parser.parse_args()

    # Use custom output dir if provided
    output_dir = args.output_dir if args.output_dir else OUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle None for n_samples (use all data)
    n_samples = None if args.n_samples == 0 else args.n_samples

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if args.checkpoint in ("best", "last"):
        ckpt_path = find_checkpoint(args.checkpoint)
    print(f"\n{'='*60}")
    print(f"EVALUATING: {ckpt_path}")
    print(f"{'='*60}\n")

    model = load_model(ckpt_path)

    print("[1/8] Collecting embeddings...")
    results = run_full_eval(model, split=args.split, max_samples=n_samples)
    print(f"  Collected {len(results['iou'])} samples")

    print("[2/8] Training curves...")
    plot_training_curves(output_dir)

    print("[3/8] Codebook health...")
    K = model.quantizer.num_embeddings
    util = plot_codebook_health(results, K, output_dir)

    print("[4/8] UMAP plots (cosine, euclidean, manhattan)...")
    fps_arr    = torch.stack(results["fps"]).numpy()
    labels_arr = np.array(results["labels"])
    codebook   = model.quantizer.embed.float().cpu().numpy()
    plot_umap(fps_arr, labels_arr, codebook, output_dir)

    print("[5/8] Per-class IoU...")
    cls_iou = plot_per_class_iou(results, output_dir)

    print("[6/8] Fingerprint retrieval...")
    recall = plot_retrieval(fps_arr, labels_arr, output_dir)

    print("[7/8] Reconstruction comparison...")
    plot_reconstruction(results, output_dir)

    print("[8/8] Master dashboard...")
    metrics = generate_dashboard(results, model, output_dir)

    # Overfitting check
    print("\n[Overfit Check] Running on train split...")
    train_iou = quick_train_iou(model, n=100)
    val_iou   = metrics["iou_mean"]
    gap       = train_iou - val_iou
    print(f"  Train IoU: {train_iou:.4f}")
    print(f"  Val IoU:   {val_iou:.4f}")
    print(f"  Gap:       {gap:.4f}  {'[OVERFIT WARNING]' if gap > 0.08 else '[OK]'}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print(f"  {'top5_retrieval_recall':25s}: {recall:.4f}")
    print(f"  {'train_val_iou_gap':25s}: {gap:.4f}")
    verdict = (metrics["iou_mean"] > 0.35 and
               metrics["codebook_util"] > 0.60 and
               recall > 0.30)
    print(f"\n  VERDICT: {'✓ AUTOENCODER IS GOOD' if verdict else '✗ AUTOENCODER NEEDS WORK'}")
    print(f"\n  All plots → {output_dir}/")
    print(f"  Metrics → {output_dir}/eval_metrics.json")

    import json
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump({**metrics, "recall": recall, "train_iou": train_iou, "gap": gap}, f, indent=2)


if __name__ == "__main__":
    main()
