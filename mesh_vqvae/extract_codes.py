"""
extract_codes.py
-----------------
One-time script: loads the best MeshGPT checkpoint, runs every sample through
encode_and_quantize(), and saves:
    trash/sedd_data/train_codes.pt  →  {"codes": [N,4096] long, "labels": [N] long}
    trash/sedd_data/val_codes.pt    →  {"codes": [M,4096] long, "labels": [M] long}

Usage:
    python extract_codes.py [--n_samples N]   # N<0 → all samples
"""

import os
import sys
import argparse
import torch
import torch.serialization
import numpy as np
from torch.utils.data import DataLoader

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from config import SmallModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset

CKPT = (
    "/data/joshi/MESHGPT/new_implementation/runs/run_20260310_040537/"
    "checkpoints/small-epoch=0070-val/iou=0.4227-val/cls_acc=0.7000.ckpt"
)
OUT_DIR = "/data/joshi/MESHGPT/new_implementation/trash/sedd_data"


@torch.no_grad()
def extract(model, loader, device, max_samples=-1):
    model.eval()
    all_codes, all_labels = [], []
    total = 0
    for batch in loader:
        pts  = batch["points"].to(device)
        nrm  = batch["normals"].to(device)
        curv = batch["curvature"].to(device)
        lbl  = batch["label"]

        enc = model.encode_and_quantize(pts, nrm, curv, lbl.to(device))
        codes = enc["code_all"].cpu().short()   # [B, 4096] — store as int16 to save space
        all_codes.append(codes)
        all_labels.append(lbl)
        total += len(codes)
        print(f"  extracted {total} samples ...", end="\r")
        if 0 < max_samples <= total:
            break
    print()
    return torch.cat(all_codes), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       type=str, default=CKPT)
    parser.add_argument("--data_dir",   type=str, default="/data/joshi/modelnet40_meshes")
    parser.add_argument("--cache_dir",  type=str,
                        default="/data/joshi/MESHGPT/new_implementation/trash/cache")
    parser.add_argument("--out_dir",    type=str, default=OUT_DIR)
    parser.add_argument("--n_samples",  type=int, default=-1,
                        help="Max samples per split. -1 = all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers",type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ── load model ────────────────────────────────────────────────────────
    cfg = SmallModelConfig()
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    model = MaskedVQVAE3D.load_from_checkpoint(args.ckpt, cfg=cfg,
                                               strict=False, map_location=device)
    torch.load = _orig
    model.eval().to(device)
    print("[INFO] MeshGPT checkpoint loaded")

    for split in ("train", "val"):
        out_path = os.path.join(args.out_dir, f"{split}_codes.pt")
        if os.path.exists(out_path):
            existing = torch.load(out_path, weights_only=False)
            print(f"[INFO] {split}: already exists ({len(existing['codes'])} samples) — skipping")
            continue

        ds = ModelNet40Dataset(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            split=split,
            use_augmentation=False,
            use_contrastive=False,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        print(f"[INFO] Extracting {split} ({len(ds)} samples) ...")
        codes, labels = extract(model, loader, device, max_samples=args.n_samples)
        torch.save({"codes": codes, "labels": labels}, out_path)
        print(f"[INFO] Saved {split}: codes={tuple(codes.shape)}, labels={tuple(labels.shape)} → {out_path}")

    print("[DONE] extraction complete")


if __name__ == "__main__":
    main()
