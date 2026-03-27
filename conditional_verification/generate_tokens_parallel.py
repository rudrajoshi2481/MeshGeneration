"""
generate_tokens_parallel.py
----------------------------
Generate token sequences from trained SEDD models using 8 GPUs in parallel (DDP).

Usage:
    torchrun --nproc_per_node=8 generate_tokens_parallel.py --mode conditional --ckpt PATH
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Add paths
DIFFUSION = "/data/joshi/tmp/MeshGeneration/diffusion_model"
MESHVQVAE = "/data/joshi/tmp/MeshGeneration/mesh_vqvae/src"
sys.path.insert(0, DIFFUSION)
sys.path.insert(0, MESHVQVAE)

from SEDD import DiscreteDiffusionTransformer
from preprocessing import MODELNET40_CLASSES

DEFAULT_OUT = "/data/joshi/tmp/MeshGeneration/runs/classifier_eval"


def setup_ddp():
    """Initialize DDP"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


@torch.no_grad()
def generate_conditional_tokens_parallel(model, device, rank, world_size, 
                                         n_samples_per_class=100, num_classes=40):
    """Generate tokens with class conditioning across multiple GPUs"""
    model.eval()
    
    # Divide classes among GPUs
    classes_per_gpu = num_classes // world_size
    start_class = rank * classes_per_gpu
    end_class = start_class + classes_per_gpu if rank < world_size - 1 else num_classes
    
    all_tokens = []
    all_labels = []
    
    if rank == 0:
        print(f"[INFO] Generating {n_samples_per_class} samples per class")
        print(f"[INFO] Total: {n_samples_per_class * num_classes} samples across {world_size} GPUs")
    
    for class_id in range(start_class, end_class):
        batch_size = 32
        n_batches = (n_samples_per_class + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            actual_batch = min(batch_size, n_samples_per_class - batch_idx * batch_size)
            
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
        
        if rank == 0:
            print(f"  GPU {rank}: class {class_id}/{end_class-1} done", end="\r")
    
    if rank == 0:
        print()
    
    return torch.cat(all_tokens), torch.cat(all_labels)


@torch.no_grad()
def generate_unconditional_tokens_parallel(model, device, rank, world_size, n_samples=4000):
    """Generate tokens without class conditioning across multiple GPUs"""
    model.eval()
    
    # Divide samples among GPUs
    samples_per_gpu = n_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else n_samples
    actual_samples = end_idx - start_idx
    
    all_tokens = []
    
    if rank == 0:
        print(f"[INFO] Generating {n_samples} unconditional samples across {world_size} GPUs")
    
    batch_size = 32
    n_batches = (actual_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        actual_batch = min(batch_size, actual_samples - batch_idx * batch_size)
        
        tokens = model.generate(
            batch_size=actual_batch,
            seq_len=4096,
            class_labels=None,
            temperature=1.0,
            num_steps=50
        )
        all_tokens.append(tokens.cpu())
        
        if rank == 0 and batch_idx % 10 == 0:
            print(f"  GPU {rank}: {batch_idx}/{n_batches} batches done", end="\r")
    
    if rank == 0:
        print()
    
    # Assign random labels for unconditional (just for compatibility)
    all_labels = torch.randint(0, 40, (actual_samples,))
    
    return torch.cat(all_tokens), all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["conditional", "unconditional"], required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SEDD checkpoint")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--n_samples_per_class", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=4000)
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Token Generation (Parallel) — {args.mode}")
        print(f"  Checkpoint: {args.ckpt}")
        print(f"  GPUs: {world_size}")
        print(f"{'='*60}\n")
    
    # Load model
    if rank == 0:
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
    
    if rank == 0:
        print(f"[INFO] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Generate tokens
    if args.mode == "conditional":
        tokens, labels = generate_conditional_tokens_parallel(
            model, device, rank, world_size, n_samples_per_class=args.n_samples_per_class
        )
    else:
        tokens, labels = generate_unconditional_tokens_parallel(
            model, device, rank, world_size, n_samples=args.n_samples
        )
    
    # Gather all tokens from all GPUs to rank 0
    if rank == 0:
        print("[INFO] Gathering tokens from all GPUs...")
    
    # Convert to list for gathering
    tokens_list = [None] * world_size
    labels_list = [None] * world_size
    
    dist.all_gather_object(tokens_list, tokens)
    dist.all_gather_object(labels_list, labels)
    
    # Save only on rank 0
    if rank == 0:
        all_tokens = torch.cat(tokens_list)
        all_labels = torch.cat(labels_list)
        
        out_path = os.path.join(args.out_dir, f"{args.mode}_tokens.pt")
        torch.save({"tokens": all_tokens.long(), "labels": all_labels.long()}, out_path)
        
        print(f"\n[DONE] Saved {len(all_tokens)} samples → {out_path}")
        print(f"       tokens shape: {tuple(all_tokens.shape)}")
        print(f"       labels shape: {tuple(all_labels.shape)}")
        
        # Stats
        if args.mode == "conditional":
            unique, counts = all_labels.unique(return_counts=True)
            print(f"       samples per class: min={counts.min()}, max={counts.max()}, mean={counts.float().mean():.1f}")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
