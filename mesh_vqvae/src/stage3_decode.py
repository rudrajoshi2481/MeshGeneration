"""
stage3_decode.py — Stage 3: SEDD-generated codes → MeshGPT decoder → reconstruction plots

Key insight (from evaluate.py): Good plots come from querying SURFACE POINTS
(the same query_pts used during training) through the decoder, not a dense grid.
This gives sparse but meaningful occupancy predictions that reveal shape structure.

Pipeline per class:
    GT path:   real surface pts → MeshGPT encoder → codes → demasker → decoder → occupancy
    SEDD path: class label → SEDD → codes → codebook lookup → demasker → decoder → occupancy
    Plot:      multi-view 2D scatter (XY, XZ, YZ) colored by GT/Pred occupancy (evaluate.py style)

Outputs → /data/joshi/MESHGPT/new_implementation/trash/stage3/
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── path setup ───────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
SEDD_DIR = "/data/joshi/yagnas_stuff/MLopsThesis/Models"
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SEDD_DIR)

from config import SmallModelConfig
from model  import MaskedVQVAE3D
from dataset import ModelNet40Dataset
from preprocessing import MODELNET40_CLASSES, CLASS_TO_IDX
from SEDD import DiscreteDiffusionTransformer

# ── paths ─────────────────────────────────────────────────────────────────────
MESHGPT_CKPT = (
    "/data/joshi/MESHGPT/new_implementation/runs/run_20260310_040537/"
    "checkpoints/small-epoch=0070-val/iou=0.4227-val/cls_acc=0.7000.ckpt"
)
SEDD_CKPT = (
    "/data/joshi/MESHGPT/new_implementation/runs/sedd_full_best/"
    "checkpoints/sedd-epoch=0069-val_loss=1.9378.ckpt"
)
DATA_DIR  = "/data/joshi/modelnet40_meshes"
CACHE_DIR = "/data/joshi/MESHGPT/new_implementation/trash/cache"
OUT_DIR   = "/data/joshi/MESHGPT/new_implementation/trash/stage3"

# 8 visually distinct classes
TARGET_CLASSES = ["airplane", "chair", "car", "sofa", "table",
                  "bathtub", "monitor", "toilet"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── model loading ─────────────────────────────────────────────────────────────
def load_meshgpt(ckpt_path):
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    model = MaskedVQVAE3D.load_from_checkpoint(ckpt_path, map_location=DEVICE)
    torch.load = _orig
    return model.eval().to(DEVICE)


def load_sedd(ckpt_path):
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    model = DiscreteDiffusionTransformer.load_from_checkpoint(ckpt_path, map_location=DEVICE)
    torch.load = _orig
    return model.eval().to(DEVICE)


# ── decode: codes → demasker → decoder → occupancy at query points ────────────
@torch.no_grad()
def codes_to_pred_occ(meshgpt, codes_1d, query_pts):
    """
    codes_1d:  [N_vox] long  — codebook indices (e.g. 4096 for 16^3 grid)
    query_pts: [M, 3]  float — surface query points in [-0.5, 0.5]

    Returns: pred_occ [M] float in [0,1]  (sigmoid of decoder logits)

    Strategy: look up codebook embeddings → treat ALL as 'kept' →
    run through demasker (with empty remain) → decoder at query points.
    """
    N_vox = codes_1d.shape[0]
    device = DEVICE

    # 1. Codebook lookup: [N_vox] → [1, N_vox, D]
    quant_all = meshgpt.quantizer.embed[codes_1d.to(device)]  # [N_vox, D]
    quant_all = quant_all.unsqueeze(0)                        # [1, N_vox, D]

    # 2. Demasker: all positions are "kept", none remain masked
    kept_idx   = torch.arange(N_vox, device=device).unsqueeze(0)         # [1, N_vox]
    remain_idx = torch.zeros(1, 0, dtype=torch.long, device=device)       # [1, 0]
    grid_mask  = torch.ones(1, N_vox, dtype=torch.bool, device=device)

    full_feat = meshgpt.demasker(quant_all, kept_idx, remain_idx, grid_mask)
    # full_feat: [1, N_vox, output_dim]

    # 3. Decoder at query points
    q = torch.from_numpy(query_pts).float().unsqueeze(0).to(device)  # [1, M, 3]
    logits = meshgpt.decoder(full_feat, q)   # [1, M]
    pred = logits.sigmoid().squeeze(0).cpu().numpy()  # [M]
    return pred


@torch.no_grad()
def gt_forward(meshgpt, sample, device=DEVICE):
    """
    Run real sample through MeshGPT encoder → decoder.
    Returns gt_occ [M], pred_occ [M], query_pts [M,3], gt_codes [N_vox]
    """
    pts = sample["points"].unsqueeze(0).to(device)
    nrm = sample["normals"].unsqueeze(0).to(device)
    cur = sample["curvature"].unsqueeze(0).to(device)
    q   = sample["query_pts"].unsqueeze(0).to(device)   # [1, M, 3]
    occ = sample["occupancy"]                           # [M] — GT labels

    enc = meshgpt.encode_and_quantize(pts, nrm, cur)
    quant_kept  = enc["quant_kept"]
    masker_out  = enc["masker_out"]
    codes_gt    = enc["code_all"].squeeze(0)  # [N_vox]

    full_feat = meshgpt.demasker(
        quant_kept,
        masker_out["sample_index"],
        masker_out["remain_index"],
        masker_out["grid_mask_flat"],
    )
    logits = meshgpt.decoder(full_feat, q)             # [1, M]
    pred   = logits.sigmoid().squeeze(0).cpu().numpy() # [M]
    gt     = occ.numpy()                               # [M]
    qpts   = q.squeeze(0).cpu().numpy()                # [M, 3]

    return gt, pred, qpts, codes_gt


# ── plotting (evaluate.py style) ──────────────────────────────────────────────
def scatter_view(ax, pts, mask, color, title, xi, yi, s=2, alpha=0.6):
    """2D scatter of pts[mask] on axes xi, yi."""
    inside = pts[mask > 0.5]
    if len(inside):
        ax.scatter(inside[:, xi], inside[:, yi], c=color, s=s, alpha=alpha,
                   linewidths=0, edgecolors='none', rasterized=True)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_aspect("equal")
    ax.axis("off")


def scatter_error(ax, pts, gt, pred, title, xi, yi, s=2):
    """Error heatmap on axes xi, yi."""
    pred_bin = (pred > 0.5).astype(float)
    error    = np.abs(pred_bin - gt)
    sc = ax.scatter(pts[:, xi], pts[:, yi], c=error,
                    cmap="RdYlGn_r", s=s, alpha=0.7, vmin=0, vmax=1,
                    linewidths=0, edgecolors='none', rasterized=True)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def compute_iou(gt, pred, thresh=0.5):
    p = (pred > thresh).astype(float)
    inter = (p * gt).sum()
    union = ((p + gt) > 0).sum()
    return float(inter / (union + 1e-8))


def code_overlap(codes_a, codes_b, vocab=256):
    ha, _ = np.histogram(codes_a.numpy(), bins=vocab, range=(0, vocab))
    hb, _ = np.histogram(codes_b.numpy(), bins=vocab, range=(0, vocab))
    ha = ha / (ha.sum() + 1e-8)
    hb = hb / (hb.sum() + 1e-8)
    return float(np.minimum(ha, hb).sum())


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Stage3] Output → {OUT_DIR}")

    # Load models
    print("[Stage3] Loading MeshGPT...")
    meshgpt = load_meshgpt(MESHGPT_CKPT)
    print(f"  Codebook: {meshgpt.quantizer.num_embeddings} × {meshgpt.quantizer.embedding_dim}")

    print("[Stage3] Loading SEDD...")
    sedd = load_sedd(SEDD_CKPT)
    print(f"  vocab={sedd.vocab_size}, seq_len={sedd.max_seq_len}, d_model={sedd.d_model}")

    # Load dataset
    print("[Stage3] Loading dataset...")
    val_ds = ModelNet40Dataset(
        data_dir=DATA_DIR, cache_dir=CACHE_DIR, split="val",
        num_surface=2048, num_query=2048,
        use_augmentation=False, use_contrastive=False,
    )

    # One sample per target class
    class_samples = {}
    for i, sample in enumerate(val_ds):
        cls = MODELNET40_CLASSES[int(sample["label"].item())]
        if cls in TARGET_CLASSES and cls not in class_samples:
            class_samples[cls] = sample
        if len(class_samples) == len(TARGET_CLASSES):
            break
    print(f"  GT samples: {list(class_samples.keys())}")

    # ── process each class ────────────────────────────────────────────────────
    results = {}
    for cls_name in TARGET_CLASSES:
        if cls_name not in class_samples:
            print(f"  [SKIP] {cls_name}")
            continue

        cls_idx = CLASS_TO_IDX[cls_name]
        print(f"\n[Stage3] {cls_name} (cls_idx={cls_idx})")

        sample = class_samples[cls_name]

        # GT forward (real encoder path)
        t0 = time.time()
        gt_occ, pred_gt, qpts, codes_gt = gt_forward(meshgpt, sample)
        iou_gt = compute_iou(gt_occ, pred_gt)
        print(f"  GT   iou={iou_gt:.3f}  occ_rate={gt_occ.mean():.3f}  [{time.time()-t0:.1f}s]")

        # SEDD generation
        t0 = time.time()
        with torch.no_grad():
            gen_codes = sedd.generate(
                batch_size=1,
                class_labels=torch.tensor([cls_idx], device=DEVICE),
                num_steps=50, temperature=1.0,
            ).squeeze(0)  # [N_vox]

        # Decode SEDD codes using same query points
        pred_sedd = codes_to_pred_occ(meshgpt, gen_codes, qpts)
        iou_sedd  = compute_iou(gt_occ, pred_sedd)
        overlap   = code_overlap(codes_gt.cpu(), gen_codes.cpu())
        print(f"  SEDD iou={iou_sedd:.3f}  overlap={overlap:.3f}  [{time.time()-t0:.1f}s]")

        results[cls_name] = dict(
            gt_occ=gt_occ, pred_gt=pred_gt, pred_sedd=pred_sedd,
            qpts=qpts, codes_gt=codes_gt.cpu(), codes_sedd=gen_codes.cpu(),
            iou_gt=iou_gt, iou_sedd=iou_sedd, overlap=overlap,
        )

    # ── PLOT 1: reconstruction comparison (evaluate.py style) ─────────────────
    # Columns: GT-XY | GT-XZ | GT-YZ | Pred(GT codes)-XY | Pred(GT codes)-XZ | Pred(GT codes)-YZ
    #          | SEDD-XY | SEDD-XZ | SEDD-YZ | Error(GT codes) | Error(SEDD)
    n = len(results)
    N_COLS = 11
    fig, axes = plt.subplots(n, N_COLS, figsize=(N_COLS * 2.2, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels = [
        "GT\n(XY)", "GT\n(XZ)", "GT\n(YZ)",
        "MeshGPT Recon\n(XY)", "MeshGPT Recon\n(XZ)", "MeshGPT Recon\n(YZ)",
        "SEDD Gen\n(XY)", "SEDD Gen\n(XZ)", "SEDD Gen\n(YZ)",
        "Error\nGT codes", "Error\nSEDD codes",
    ]
    for col_i, lbl in enumerate(col_labels):
        axes[0, col_i].set_title(lbl, fontsize=8, pad=4, weight='bold')

    fig.suptitle(
        "Stage 3 — Reconstruction Comparison  |  "
        "GT (blue) | MeshGPT Recon (green) | SEDD Gen (red) | Error=|pred−gt| (RdYlGn)",
        fontsize=11, weight='bold', y=1.005
    )

    for row, (cls_name, r) in enumerate(results.items()):
        qpts   = r["qpts"]
        gt     = r["gt_occ"]
        p_gt   = r["pred_gt"]
        p_sedd = r["pred_sedd"]

        # Row label
        axes[row, 0].annotate(
            f"{cls_name.upper()}\nIoU(GT)={r['iou_gt']:.3f}\nIoU(SEDD)={r['iou_sedd']:.3f}",
            xy=(-0.35, 0.5), xycoords='axes fraction', fontsize=7,
            va='center', ha='right', weight='bold',
        )

        # GT: XY, XZ, YZ
        scatter_view(axes[row, 0],  qpts, gt, 'royalblue',    '', 0, 1)
        scatter_view(axes[row, 1],  qpts, gt, 'royalblue',    '', 0, 2)
        scatter_view(axes[row, 2],  qpts, gt, 'royalblue',    '', 1, 2)

        # MeshGPT recon (GT codes): XY, XZ, YZ
        scatter_view(axes[row, 3],  qpts, p_gt, 'mediumseagreen', '', 0, 1)
        scatter_view(axes[row, 4],  qpts, p_gt, 'mediumseagreen', '', 0, 2)
        scatter_view(axes[row, 5],  qpts, p_gt, 'mediumseagreen', '', 1, 2)

        # SEDD generated: XY, XZ, YZ
        scatter_view(axes[row, 6],  qpts, p_sedd, 'tomato',  '', 0, 1)
        scatter_view(axes[row, 7],  qpts, p_sedd, 'tomato',  '', 0, 2)
        scatter_view(axes[row, 8],  qpts, p_sedd, 'tomato',  '', 1, 2)

        # Error plots (YZ view)
        scatter_error(axes[row, 9],  qpts, gt, p_gt,   '', 1, 2)
        sc = scatter_error(axes[row, 10], qpts, gt, p_sedd, '', 1, 2)

        if row == 0:
            plt.colorbar(sc, ax=axes[row, 10], fraction=0.06, pad=0.08, label="Error")

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    p1 = os.path.join(OUT_DIR, "stage3_reconstruction.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Stage3] Saved reconstruction → {p1}")

    # ── PLOT 2: code distribution per class ────────────────────────────────────
    vocab = meshgpt.quantizer.num_embeddings
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig2, axes2 = plt.subplots(n_rows * 2, n_cols,
                                figsize=(n_cols * 4, n_rows * 4))
    fig2.suptitle("Code Distributions: GT Codes (blue) vs SEDD-Generated (red)",
                  fontsize=13, weight='bold')

    for i, (cls_name, r) in enumerate(results.items()):
        row_gt  = (i // n_cols) * 2
        row_gen = row_gt + 1
        col     = i % n_cols

        ax_gt  = axes2[row_gt,  col]
        ax_gen = axes2[row_gen, col]

        gt_flat  = r["codes_gt"].flatten().numpy()
        gen_flat = r["codes_sedd"].flatten().numpy()

        ax_gt.bar(np.arange(vocab),
                  np.bincount(gt_flat.astype(int), minlength=vocab),
                  width=1.0, color='royalblue', linewidth=0, alpha=0.8)
        ax_gt.set_title(f"{cls_name} — GT codes", fontsize=9, weight='bold')
        ax_gt.set_xlabel("Code ID", fontsize=7)
        ax_gt.set_ylabel("Count", fontsize=7)
        ax_gt.grid(axis='y', alpha=0.3)

        ax_gen.bar(np.arange(vocab),
                   np.bincount(gen_flat.astype(int), minlength=vocab),
                   width=1.0, color='tomato', linewidth=0, alpha=0.8)
        ax_gen.set_title(f"{cls_name} — SEDD codes  (overlap={r['overlap']:.3f})", fontsize=9, weight='bold')
        ax_gen.set_xlabel("Code ID", fontsize=7)
        ax_gen.set_ylabel("Count", fontsize=7)
        ax_gen.grid(axis='y', alpha=0.3)

    for j in range(i + 1, n_rows * n_cols):
        axes2[(j // n_cols) * 2, j % n_cols].axis('off')
        axes2[(j // n_cols) * 2 + 1, j % n_cols].axis('off')

    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "stage3_code_distributions.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Stage3] Saved code distributions → {p2}")

    # ── REPORT ────────────────────────────────────────────────────────────────
    summary = {
        cls: {
            "iou_gt":    round(r["iou_gt"],    4),
            "iou_sedd":  round(r["iou_sedd"],  4),
            "code_overlap": round(r["overlap"], 4),
        }
        for cls, r in results.items()
    }
    iou_gts   = [r["iou_gt"]   for r in results.values()]
    iou_sedds = [r["iou_sedd"] for r in results.values()]
    overlaps  = [r["overlap"]  for r in results.values()]
    summary["_mean"] = {
        "iou_gt":     round(float(np.mean(iou_gts)),   4),
        "iou_sedd":   round(float(np.mean(iou_sedds)), 4),
        "code_overlap": round(float(np.mean(overlaps)), 4),
    }

    with open(os.path.join(OUT_DIR, "stage3_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("  CLASS            IoU(GT)  IoU(SEDD)  Code Overlap")
    print("  " + "-"*55)
    for cls, r in results.items():
        print(f"  {cls:15s}  {r['iou_gt']:.3f}    {r['iou_sedd']:.3f}      {r['overlap']:.3f}")
    print("  " + "-"*55)
    print(f"  {'MEAN':15s}  {np.mean(iou_gts):.3f}    {np.mean(iou_sedds):.3f}      {np.mean(overlaps):.3f}")
    print("="*60)
    print(f"\n  Plots: {OUT_DIR}")
    print(f"  stage3_reconstruction.png")
    print(f"  stage3_code_distributions.png")
    print(f"  stage3_report.json")


if __name__ == "__main__":
    main()
