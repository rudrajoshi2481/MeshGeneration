"""
decode_generated_tokens.py
--------------------------
Decode generated token sequences back to 3D meshes using VQVAE decoder.

Verifies that generated tokens produce valid 3D shapes.

Usage:
    # Decode SEDD generated tokens
    python decode_generated_tokens.py \
        --tokens_path trash/sedd_generated/tokens.pt \
        --vqvae_ckpt trash/vqvae/checkpoints/best.ckpt \
        --out_dir trash/decoded_meshes \
        --n_samples 10

    # Decode DoT generated tokens
    python decode_generated_tokens.py \
        --tokens_path trash/dot_generated/tokens.pt \
        --vqvae_ckpt trash/vqvae/checkpoints/best.ckpt \
        --out_dir trash/decoded_meshes \
        --n_samples 10
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
VQVAE = os.path.join(_BASE, "mesh_vqvae", "src")
sys.path.insert(0, VQVAE)

from model import MaskedVQVAE3D
from config import SmallModelConfig
from preprocessing import MODELNET40_CLASSES

# ── constants ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vqvae(ckpt_path: str):
    """Load trained VQVAE model."""
    print(f"[INFO] Loading VQVAE from: {ckpt_path}")
    
    cfg = SmallModelConfig()
    model = MaskedVQVAE3D.load_from_checkpoint(
        ckpt_path, 
        cfg=cfg, 
        strict=False, 
        map_location=DEVICE
    )
    model.eval().to(DEVICE)
    print(f"[INFO] VQVAE loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


def load_tokens(tokens_path: str, n_samples: int = -1):
    """Load generated token sequences."""
    print(f"[INFO] Loading tokens from: {tokens_path}")
    
    data = torch.load(tokens_path, map_location="cpu", weights_only=False)
    tokens = data.get("tokens", data.get("codes")).long()
    labels = data.get("labels", torch.zeros(len(tokens)))
    
    if n_samples > 0 and n_samples < len(tokens):
        indices = torch.randperm(len(tokens))[:n_samples]
        tokens = tokens[indices]
        labels = labels[indices]
    
    print(f"[INFO] Loaded {len(tokens)} samples, shape: {tokens.shape}")
    print(f"[INFO] Token range: [{tokens.min()}, {tokens.max()}]")
    return tokens, labels


@torch.no_grad()
def decode_tokens(model, tokens, labels, batch_size=8):
    """Decode token sequences to point clouds."""
    print(f"[INFO] Decoding {len(tokens)} meshes...")
    
    all_points = []
    all_occupancy = []
    
    for i in range(0, len(tokens), batch_size):
        batch_tokens = tokens[i:i+batch_size].to(DEVICE)
        batch_labels = labels[i:i+batch_size].to(DEVICE) if labels.numel() > 0 else None
        
        # Decode: tokens → embeddings → point clouds
        decoded = model.decode_from_codes(batch_tokens)
        
        points = decoded["points"].cpu()
        occupancy = decoded.get("occupancy", torch.ones(points.shape[0], points.shape[1]))
        
        all_points.append(points)
        all_occupancy.append(occupancy.cpu())
        
        print(f"  decoded {min(i+batch_size, len(tokens))}/{len(tokens)}...")
    
    points = torch.cat(all_points, dim=0)
    occupancy = torch.cat(all_occupancy, dim=0)
    
    print(f"[INFO] Decoding complete. Output shape: {points.shape}")
    return points, occupancy


def visualize_mesh(points, occupancy, class_name, idx, out_dir):
    """Save 3D visualization of decoded mesh."""
    # Filter by occupancy
    occupied_mask = occupancy > 0.5
    if occupied_mask.dim() == 1:
        occupied_points = points[occupied_mask]
    else:
        occupied_points = points
    
    if len(occupied_points) == 0:
        print(f"[WARN] No occupied points for sample {idx}")
        return
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization (max 2000)
    n_vis = min(2000, len(occupied_points))
    vis_points = occupied_points[torch.randperm(len(occupied_points))[:n_vis]]
    
    ax.scatter(
        vis_points[:, 0], 
        vis_points[:, 1], 
        vis_points[:, 2],
        c=vis_points[:, 2],  # Color by Z
        cmap='viridis',
        s=5,
        alpha=0.6
    )
    
    ax.set_title(f"{class_name} (Sample {idx})", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Equal aspect ratio
    max_range = vis_points.max() - vis_points.min()
    mid_x = (vis_points[:, 0].max() + vis_points[:, 0].min()) * 0.5
    mid_y = (vis_points[:, 1].max() + vis_points[:, 1].min()) * 0.5
    mid_z = (vis_points[:, 2].max() + vis_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    save_path = os.path.join(out_dir, f"mesh_{idx:03d}_{class_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def save_point_cloud(points, occupancy, class_name, idx, out_dir):
    """Save point cloud as .ply file."""
    try:
        import open3d as o3d
        
        # Filter occupied points
        occupied_mask = occupancy > 0.5
        if occupied_mask.dim() == 1:
            occupied_points = points[occupied_mask].numpy()
        else:
            occupied_points = points.numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occupied_points)
        
        save_path = os.path.join(out_dir, f"mesh_{idx:03d}_{class_name}.ply")
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"[INFO] Saved PLY: {save_path}")
    except ImportError:
        print("[WARN] open3d not installed, skipping PLY export")


def main():
    parser = argparse.ArgumentParser(description="Decode token sequences to 3D meshes")
    parser.add_argument("--tokens_path", type=str, required=True,
                        help="Path to generated tokens .pt file")
    parser.add_argument("--vqvae_ckpt", type=str, required=True,
                        help="Path to VQVAE checkpoint")
    parser.add_argument("--out_dir", type=str, default="trash/decoded_meshes",
                        help="Output directory for visualizations")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to decode (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for decoding")
    parser.add_argument("--save_ply", action="store_true",
                        help="Also save as .ply point cloud files")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Decode Generated Tokens → 3D Meshes")
    print(f"{'='*60}\n")
    
    # Load models and data
    vqvae = load_vqvae(args.vqvae_ckpt)
    tokens, labels = load_tokens(args.tokens_path, args.n_samples)
    
    # Decode
    points, occupancy = decode_tokens(vqvae, tokens, labels, args.batch_size)
    
    # Visualize each mesh
    print(f"\n[INFO] Generating visualizations...")
    for i in range(len(points)):
        class_idx = int(labels[i].item()) if labels.numel() > 0 else 0
        class_name = MODELNET40_CLASSES[class_idx] if class_idx < len(MODELNET40_CLASSES) else f"class_{class_idx}"
        
        visualize_mesh(points[i], occupancy[i], class_name, i, args.out_dir)
        
        if args.save_ply:
            save_point_cloud(points[i], occupancy[i], class_name, i, args.out_dir)
    
    print(f"\n[DONE] Decoded {len(points)} meshes to: {args.out_dir}/")
    print(f"[INFO] View PNG files to verify mesh quality")


if __name__ == "__main__":
    main()
