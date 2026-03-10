"""
test_overfit.py — Overfit test on a single batch.
Verifies: forward pass works, loss decreases, shapes are correct.
Saves: loss curve, codebook utilization plot.

Usage:
    python test_overfit.py
Output goes to: /data/joshi/MESHGPT/new_implementation/trash/overfit/
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from config import SmallModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset

OUTPUT_DIR = "/data/joshi/MESHGPT/new_implementation/trash/overfit"
CACHE_DIR  = "/data/joshi/MESHGPT/new_implementation/trash/cache"
DATA_DIR   = "/data/joshi/modelnet40_meshes"
DEVICE     = "cuda:0"
N_STEPS    = 300
BATCH_SIZE = 8


def build_single_batch(device):
    """Load a tiny fixed batch (8 samples) and pin it to GPU."""
    print("[Overfit] Loading dataset...")
    ds = ModelNet40Dataset(
        data_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
        split="train",
        num_surface=2048,
        num_query=2048,
        use_augmentation=False,
        use_contrastive=True,
        seed=42,
    )
    batch_list = [ds[i] for i in range(BATCH_SIZE)]

    def stack(key):
        return torch.stack([b[key] for b in batch_list]).to(device)

    batch = {k: stack(k) for k in batch_list[0] if isinstance(batch_list[0][k], torch.Tensor)}
    return batch


def run_overfit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = SmallModelConfig()
    cfg.train.cache_dir = CACHE_DIR

    print("[Overfit] Building model...")
    model = MaskedVQVAE3D(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Overfit] Parameters: {n_params:.2f}M")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = build_single_batch(DEVICE)
    print("[Overfit] Batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    # ---- Verify forward pass shapes ----
    print("\n[Overfit] Running test forward pass...")
    with torch.no_grad():
        out = model(
            batch["points"], batch["normals"], batch["curvature"],
            batch["query_pts"], batch["label"],
        )
    print("[Overfit] Output shapes:")
    print(f"  logits:       {tuple(out['logits'].shape)}")
    print(f"  class_logits: {tuple(out['class_logits'].shape)}")
    print(f"  fingerprint:  {tuple(out['fingerprint'].shape)}")
    print(f"  code_indices: {tuple(out['code_indices'].shape)}")
    print(f"  vq_loss:      {out['vq_loss'].item():.4f}")

    util = model.quantizer.codebook_utilization(out["code_indices"])
    print(f"  codebook_util: {util:.3f}")

    # ---- Overfit loop ----
    print(f"\n[Overfit] Training on fixed batch for {N_STEPS} steps...")

    loss_history = []
    recon_history = []
    util_history = []

    for step in range(N_STEPS):
        model._step = step + 1
        optimizer.zero_grad()

        out = model(
            batch["points"], batch["normals"], batch["curvature"],
            batch["query_pts"], batch["label"],
        )

        # Losses
        recon_loss = F.binary_cross_entropy_with_logits(out["logits"], batch["occupancy"])
        cls_w = min(0.5, 0.5 * (step + 1) / 5000)
        cls_loss = F.cross_entropy(out["class_logits"], batch["label"])
        total = recon_loss + cfg.train.vq_beta * out["vq_loss"] + cls_w * cls_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        util = model.quantizer.codebook_utilization(out["code_indices"])
        loss_history.append(total.item())
        recon_history.append(recon_loss.item())
        util_history.append(util)

        if step % 50 == 0 or step == N_STEPS - 1:
            acc = (out["class_logits"].argmax(1) == batch["label"]).float().mean().item()
            print(f"  step {step:4d}  total={total.item():.4f}  "
                  f"recon={recon_loss.item():.4f}  "
                  f"vq={out['vq_loss'].item():.4f}  "
                  f"cls={cls_loss.item():.4f}  "
                  f"util={util:.3f}  "
                  f"cls_acc={acc:.3f}")

    # ---- Verify IoU after overfitting ----
    model.eval()
    with torch.no_grad():
        out = model(
            batch["points"], batch["normals"], batch["curvature"],
            batch["query_pts"], batch["label"],
        )
    probs = out["logits"].sigmoid()
    preds = (probs > 0.5).float()
    gt = batch["occupancy"]
    # Diagnostics
    pred_mean = probs.mean().item()
    gt_mean = gt.mean().item()
    print(f"\n[Overfit] Prediction stats: mean_prob={pred_mean:.3f}, gt_mean={gt_mean:.3f}")
    print(f"[Overfit] Pred range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")

    iou = MaskedVQVAE3D._batch_iou(preds, gt)
    # If preds are all-1 (trivial), iou = gt_mean which may be 0.3-0.5
    # A meaningful overfit: model should match gt better than all-1 baseline
    baseline_iou = MaskedVQVAE3D._batch_iou(torch.ones_like(gt), gt)
    print(f"[Overfit] Final IoU on fixed batch: {iou.item():.4f}")
    print(f"[Overfit] Baseline (all-1) IoU:     {baseline_iou.item():.4f}")
    print(f"[Overfit] Final recon loss: {recon_history[-1]:.4f}")
    print(f"[Overfit] Final codebook util: {util_history[-1]:.3f}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(loss_history, label="total", color="blue")
    axes[0].plot(recon_history, label="recon", color="orange")
    axes[0].set_title("Overfit: Loss Curves")
    axes[0].set_xlabel("Step")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(util_history, color="green")
    axes[1].set_title("Codebook Utilization")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Fraction Active")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    # Codebook usage histogram
    with torch.no_grad():
        out = model(
            batch["points"], batch["normals"], batch["curvature"],
            batch["query_pts"], batch["label"],
        )
    K = model.quantizer.num_embeddings
    counts = torch.bincount(out["code_indices"].flatten().cpu(), minlength=K).numpy()
    axes[2].bar(range(K), counts, color="steelblue", width=1.0)
    active = (counts > 0).sum()
    axes[2].set_title(f"Codebook Usage: {active}/{K} active ({100*active/K:.1f}%)")
    axes[2].set_xlabel("Code Index")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "overfit_results.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\n[Overfit] Plot saved → {plot_path}")

    # ---- Assertion summary ----
    print("\n[Overfit] === PASS/FAIL SUMMARY ===")
    checks = {
        "Loss decreased (total)": loss_history[-1] < loss_history[0],
        "Loss decreased (recon)": recon_history[-1] < recon_history[0],
        "Codebook utilization > 0": util_history[-1] > 0,
        "IoU >= baseline (all-1)": iou.item() >= baseline_iou.item() - 0.01,
        "Final recon < 0.5": recon_history[-1] < 0.5,
    }
    all_pass = True
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n[Overfit] Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


if __name__ == "__main__":
    run_overfit()
