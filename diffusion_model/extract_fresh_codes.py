"""
extract_fresh_codes.py
-----------------------
Extract code sequences from the latest mesh_vqvae checkpoint and save to
/data/joshi/tmp/MeshGeneration/runs/sedd_data/

Usage:
    python extract_fresh_codes.py [--ckpt PATH] [--out_dir DIR]
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

SRC = "/data/joshi/tmp/MeshGeneration/mesh_vqvae/src"
sys.path.insert(0, SRC)

from config import SmallModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset

DEFAULT_CKPT = (
    "/data/joshi/tmp/MeshGeneration/runs/small_run/checkpoints/"
    "small-epoch=0056-val/iou=0.4227-val/cls_acc=0.6229.ckpt"
)
DEFAULT_OUT  = "/data/joshi/tmp/MeshGeneration/runs/sedd_data"


@torch.no_grad()
def extract(model, loader, device, max_samples=-1):
    model.eval()
    all_codes, all_labels = [], []
    total = 0
    for batch in loader:
        pts  = batch["points"].to(device)
        nrm  = batch["normals"].to(device)
        curv = batch["curvature"].to(device)
        lbl  = batch["label"].to(device)

        enc   = model.encode_and_quantize(pts, nrm, curv, lbl)
        codes = enc["code_all"].cpu().short()   # [B, 4096] as int16
        all_codes.append(codes)
        all_labels.append(batch["label"])
        total += len(codes)
        print(f"  extracted {total} samples ...", end="\r")
        if 0 < max_samples <= total:
            break
    print()
    return torch.cat(all_codes), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        type=str, default=DEFAULT_CKPT)
    parser.add_argument("--data_dir",    type=str, default="/data/joshi/modelnet40_meshes")
    parser.add_argument("--cache_dir",   type=str, default="/data/joshi/tmp/MeshGeneration/runs/cache")
    parser.add_argument("--out_dir",     type=str, default=DEFAULT_OUT)
    parser.add_argument("--n_samples",   type=int, default=-1)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite",   action="store_true",
                        help="Re-extract even if files already exist")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device  = {device}")
    print(f"[INFO] ckpt    = {args.ckpt}")
    print(f"[INFO] out_dir = {args.out_dir}")

    # ── load model ────────────────────────────────────────────────────────
    cfg = SmallModelConfig()
    # Patch torch.load to allow weights_only=False (required for PL checkpoints with numpy scalars)
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        model = MaskedVQVAE3D.load_from_checkpoint(
            args.ckpt, cfg=cfg, strict=False, map_location=device,
        )
    finally:
        torch.load = _orig_load
    model.eval().to(device)
    print("[INFO] Checkpoint loaded")

    for split in ("train", "val"):
        out_path = os.path.join(args.out_dir, f"{split}_codes.pt")
        if os.path.exists(out_path) and not args.overwrite:
            existing = torch.load(out_path, weights_only=False)
            print(f"[INFO] {split}: exists ({len(existing['codes'])} samples) — skip. Use --overwrite to re-extract")
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
        print(f"[INFO] Saved → {out_path}  shape={tuple(codes.shape)}")

    print("[DONE] extraction complete")


if __name__ == "__main__":
    main()
