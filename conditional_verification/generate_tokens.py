"""
generate_tokens.py
------------------
Generate token sequences from trained SEDD models (conditional + unconditional)
to test if class conditioning produces class-consistent tokens.

Usage:
    python generate_tokens.py --mode conditional --ckpt PATH --out_dir DIR
    python generate_tokens.py --mode unconditional --ckpt PATH --out_dir DIR
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add paths
DIFFUSION = "/data/joshi/tmp/MeshGeneration/diffusion_model"
MESHVQVAE = "/data/joshi/tmp/MeshGeneration/mesh_vqvae/src"
sys.path.insert(0, DIFFUSION)
sys.path.insert(0, MESHVQVAE)

from SEDD import DiscreteDiffusionTransformer
from preprocessing import MODELNET40_CLASSES

DEFAULT_OUT = "/data/joshi/tmp/MeshGeneration/runs/classifier_eval"


@torch.no_grad()
def generate_conditional_tokens(model, device, n_samples_per_class=100, num_classes=40):
    """Generate tokens with class conditioning"""
    model.eval()
    all_tokens = []
    all_labels = []
    
    print(f"[INFO] Generating {n_samples_per_class} samples per class (total: {n_samples_per_class * num_classes})")
    
    for class_id in tqdm(range(num_classes), desc="Generating conditional"):
        # Generate batch for this class
        batch_size = min(32, n_samples_per_class)
        n_batches = (n_samples_per_class + batch_size - 1) // batch_size
        
        for _ in range(n_batches):
            actual_batch = min(batch_size, n_samples_per_class - len(all_tokens) // 4096 % n_samples_per_class)
            if actual_batch <= 0:
                break
                
            class_labels = torch.full((actual_batch,), class_id, dtype=torch.long, device=device)
            tokens = model.generate(
                batch_size=actual_batch,
                seq_len=4096,
                class_labels=class_labels,
                temperature=1.0,
                num_steps=50
            )
            all_tokens.append(tokens.cpu())
            all_labels.append(class_labels.cpu())
    
    return torch.cat(all_tokens), torch.cat(all_labels)


@torch.no_grad()
def generate_unconditional_tokens(model, device, n_samples=4000):
    """Generate tokens without class conditioning"""
    model.eval()
    all_tokens = []
    
    print(f"[INFO] Generating {n_samples} unconditional samples")
    
    batch_size = 32
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for _ in tqdm(range(n_batches), desc="Generating unconditional"):
        actual_batch = min(batch_size, n_samples - len(all_tokens) * 4096 // 4096)
        if actual_batch <= 0:
            break
            
        tokens = model.generate(
            batch_size=actual_batch,
            seq_len=4096,
            class_labels=None,
            temperature=1.0,
            num_steps=50
        )
        all_tokens.append(tokens.cpu())
    
    # Assign random labels for unconditional (just for compatibility)
    all_labels = torch.randint(0, 40, (len(all_tokens) * batch_size,))[:n_samples]
    
    return torch.cat(all_tokens), all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["conditional", "unconditional"], required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SEDD checkpoint")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--n_samples_per_class", type=int, default=100,
                        help="Samples per class for conditional mode")
    parser.add_argument("--n_samples", type=int, default=4000,
                        help="Total samples for unconditional mode")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"  Token Generation — {args.mode}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    print("[INFO] Loading SEDD checkpoint...")
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        model = DiscreteDiffusionTransformer.load_from_checkpoint(
            args.ckpt, map_location=device
        )
    finally:
        torch.load = _orig_load
    
    model.eval().to(device)
    print(f"[INFO] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Generate tokens
    if args.mode == "conditional":
        tokens, labels = generate_conditional_tokens(
            model, device, n_samples_per_class=args.n_samples_per_class
        )
    else:
        tokens, labels = generate_unconditional_tokens(
            model, device, n_samples=args.n_samples
        )
    
    # Save
    out_path = os.path.join(args.out_dir, f"{args.mode}_tokens.pt")
    torch.save({"tokens": tokens.long(), "labels": labels.long()}, out_path)
    
    print(f"\n[DONE] Saved {len(tokens)} samples → {out_path}")
    print(f"       tokens shape: {tuple(tokens.shape)}")
    print(f"       labels shape: {tuple(labels.shape)}")
    
    # Stats
    if args.mode == "conditional":
        unique, counts = labels.unique(return_counts=True)
        print(f"       samples per class: min={counts.min()}, max={counts.max()}, mean={counts.float().mean():.1f}")


if __name__ == "__main__":
    main()
