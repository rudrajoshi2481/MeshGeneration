"""
train.py
---------
Trains ConditionalSEDD (class-as-first-token) on pre-extracted MeshGPT codes.

Usage:
    # Small test (fast, sanity check)
    python train.py --mode small

    # Full training
    python train.py --mode full

Outputs → /data/joshi/MESHGPT/trash/conditional_sedd/runs/<run_id>/
    checkpoints/
    logs/
    report.json
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from conditional_SEDD import ConditionalSEDD, class_to_token, token_to_class, MASK_TOKEN_ID, NUM_CLASSES
from dataset import ClassPrefixedCodeDataset

DATA_DIR = "/data/joshi/MESHGPT/new_implementation/trash/sedd_data"
OUT_BASE  = "/data/joshi/MESHGPT/trash/conditional_sedd/runs"

MODELNET40_CLASSES = [
    'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car',
    'chair','cone','cup','curtain','desk','door','dresser','flower_pot',
    'glass_box','guitar','keyboard','lamp','laptop','mantel','monitor',
    'night_stand','person','piano','plant','radio','range_hood','sink',
    'sofa','stairs','stool','table','tent','toilet','tv_stand','vase',
    'wardrobe','xbox'
]

# ─────────────────────────────────────────────────────────────────────────────
# Model configs
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "small": dict(
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        dropout=0.1, num_timesteps=1000, schedule_type="cosine",
        learning_rate=5e-4, beta1=0.9, beta2=0.99, weight_decay=0.01,
        batch_size=32, max_epochs=30, n_train=500, n_val=100,
        num_workers=2,
    ),
    "medium": dict(
        d_model=256, nhead=4, num_layers=4, dim_feedforward=1024,
        dropout=0.1, num_timesteps=1000, schedule_type="cosine",
        learning_rate=5e-4, beta1=0.9, beta2=0.99, weight_decay=0.01,
        batch_size=32, max_epochs=60, n_train=-1, n_val=-1,
        num_workers=4,
    ),
    "full": dict(
        d_model=512, nhead=8, num_layers=6, dim_feedforward=2048,
        dropout=0.1, num_timesteps=1000, schedule_type="cosine",
        learning_rate=3e-4, beta1=0.9, beta2=0.99, weight_decay=0.01,
        batch_size=64, max_epochs=100, n_train=-1, n_val=-1,
        num_workers=4,
        num_gpus=1, strategy="auto",
    ),
    "ddp8": dict(
        d_model=512, nhead=8, num_layers=6, dim_feedforward=2048,
        dropout=0.1, num_timesteps=1000, schedule_type="cosine",
        # effective batch = 8 GPUs x 64 x 4 accum = 2048 per update
        learning_rate=3e-4, beta1=0.9, beta2=0.99, weight_decay=0.01,
        batch_size=64, max_epochs=300, n_train=-1, n_val=-1,
        num_workers=4,
        num_gpus=8, strategy="ddp_find_unused_parameters_false",
        accumulate_grad_batches=4,  # simulate larger batch
    ),
}


def make_run_dir(mode: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_BASE, f"cond_sedd_{mode}_{ts}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


@torch.no_grad()
def evaluate_generation(model: ConditionalSEDD, run_dir: str, n_per_class: int = 5):
    """
    Generate n_per_class meshes per class and print class token statistics.
    Saves a summary to run_dir/generation_sample.json
    """
    model.eval()
    device = next(model.parameters()).device

    results = []
    for cls_id in range(NUM_CLASSES):
        class_ids = torch.tensor([cls_id] * n_per_class, device=device)
        seqs = model.generate(class_ids, temperature=1.0, num_steps=50)  # [n, 4097]

        # Check class token at position 0 is correct
        class_token_correct = (seqs[:, 0] == class_ids + 257).all().item()

        # Mesh codes are positions 1:
        mesh_codes = seqs[:, 1:]  # [n, 4096]
        unique_codes = mesh_codes.unique().numel()
        min_code = int(mesh_codes.min())
        max_code = int(mesh_codes.max())

        results.append({
            "class_id": cls_id,
            "class_name": MODELNET40_CLASSES[cls_id],
            "class_token_correct": class_token_correct,
            "unique_mesh_codes": unique_codes,
            "mesh_code_range": [min_code, max_code],
        })
        print(f"  [{cls_id:2d}] {MODELNET40_CLASSES[cls_id]:15s} | "
              f"class_token_ok={class_token_correct} | "
              f"unique_codes={unique_codes:3d} | "
              f"code_range=[{min_code:3d}, {max_code:3d}]")

    out_path = os.path.join(run_dir, "generation_sample.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Generation samples saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["small", "medium", "full", "ddp8"], default="small")
    args = parser.parse_args()

    cfg     = CONFIGS[args.mode]
    run_dir = make_run_dir(args.mode)

    print("=" * 60)
    print(f"  Conditional SEDD — mode={args.mode}")
    print(f"  run_dir = {run_dir}")
    print(f"  Sequence format: [class_token | 4096 mesh codes]")
    print(f"  Vocab size: 297 (256 codes + 1 mask + 40 class tokens)")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = ClassPrefixedCodeDataset(
        os.path.join(DATA_DIR, "train_codes.pt"),
        n_samples=cfg["n_train"],
    )
    val_ds = ClassPrefixedCodeDataset(
        os.path.join(DATA_DIR, "val_codes.pt"),
        n_samples=cfg["n_val"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    print(f"[INFO] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"[INFO] Batch size: {cfg['batch_size']} | Epochs: {cfg['max_epochs']}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalSEDD(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        num_timesteps=cfg["num_timesteps"],
        schedule_type=cfg["schedule_type"],
        learning_rate=cfg["learning_rate"],
        beta1=cfg["beta1"],
        beta2=cfg["beta2"],
        weight_decay=cfg["weight_decay"],
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Model parameters: {total_params:.2f}M")
    print(f"[INFO] Sequence length: {model.seq_len} (1 class token + 4096 mesh codes)")
    print(f"[INFO] Vocab size: {model.vocab_size} (0-255 codes, 256 mask, 257-296 class tokens)")

    # ── Tensor core optimization ──────────────────────────────────────────────
    torch.set_float32_matmul_precision('high')

    # ── GPU setup (must be before callbacks) ───────────────────────────────
    n_gpus    = cfg.get("num_gpus", 1)
    strategy  = cfg.get("strategy", "auto")
    n_devices = min(n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    accel     = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] GPUs available: {torch.cuda.device_count()} | Using: {n_devices} | Strategy: {strategy}")

    # ── Callbacks ───────────────────────────────────────────────────────────
    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="cond_sedd-{epoch:04d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    # EarlyStopping: only use in single-GPU mode (DDP per-rank early stopping is unstable)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
    ) if n_devices == 1 else None
    callbacks = [ckpt_callback] + ([early_stop] if early_stop else [])

    # ── Trainer ───────────────────────────────────────────────────────────────

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator=accel,
        devices=n_devices,
        strategy=strategy,
        precision="bf16",
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=2,
        log_every_n_steps=5,
        enable_progress_bar=True,
        default_root_dir=run_dir,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_time = (time.time() - t0) / 60.0

    print(f"\n[INFO] Training done in {train_time:.1f} min")
    print(f"[INFO] Best val_loss: {ckpt_callback.best_model_score:.4f}")
    print(f"[INFO] Best ckpt: {ckpt_callback.best_model_path}")

    # ── Post-training: only rank 0 does generation + report ──────────────────
    if trainer.is_global_zero:
        print("\n[INFO] Running generation evaluation...")
        best_ckpt = ckpt_callback.best_model_path
        if best_ckpt and os.path.exists(best_ckpt):
            best_model = ConditionalSEDD.load_from_checkpoint(best_ckpt)
            best_model.eval()
            if torch.cuda.is_available():
                best_model = best_model.cuda()
            evaluate_generation(best_model, run_dir, n_per_class=3)
        else:
            print(f"[WARN] Best checkpoint not found: {best_ckpt}")

        report = {
            "mode": args.mode,
            "config": {k: v for k, v in cfg.items()},
            "results": {
                "best_val_loss": float(ckpt_callback.best_model_score),
                "total_params_M": round(total_params, 3),
                "train_epochs": trainer.current_epoch,
                "train_time_min": round(train_time, 1),
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_gpus": n_devices,
                "effective_batch": cfg["batch_size"] * n_devices,
            },
            "model": {
                "seq_len": model.seq_len,
                "vocab_size": model.vocab_size,
                "class_token_range": [257, 296],
                "mask_token": 256,
            },
            "best_ckpt": ckpt_callback.best_model_path,
            "timestamp": datetime.now().isoformat(),
        }

        report_path = os.path.join(run_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Report saved → {report_path}")
        print("\n[DONE]")


if __name__ == "__main__":
    main()
