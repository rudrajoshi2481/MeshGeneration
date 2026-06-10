"""
test_inference.py
-----------------
Test inference modes: class ID generation and puncture refill for SEDD and DoT.

This tests the actual semantic channel functionality:
1. Generate tokens from class labels only
2. Puncture + refill recovery

Usage:
    python test_inference.py \
        --sedd_ckpt trash/sedd_runs/<run>/checkpoints/best.ckpt \
        --dot_ckpt trash/dot_runs/<run>/checkpoints/best.ckpt \
        --out_dir trash/test_inference
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import json
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "models", "diffusion"))
sys.path.insert(0, os.path.join(_BASE, "models", "autoregressive"))
sys.path.insert(0, os.path.join(_BASE, "mesh_vqvae", "src"))

from SEDD import DiscreteDiffusionTransformer
from preprocessing import MODELNET40_CLASSES

# ── constants ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 256
SEQ_LEN = 4096
NUM_CLASSES = 40
MASK_ID = 256
BOS_TOKEN = 256  # DoT uses BOS at vocab size


def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


@torch.no_grad()
def generate_sedd_class_only(sedd_model, n_per_class=2, num_steps=20):
    """Generate tokens from class labels only (no partial sequence)."""
    log(f"SEDD: Generating {n_per_class} samples per class from class labels only...")
    sedd_model.eval().to(DEVICE)
    
    all_tokens = []
    all_labels = []
    
    for class_idx in range(min(5, NUM_CLASSES)):  # Test first 5 classes
        class_labels = torch.full((n_per_class,), class_idx, device=DEVICE)
        
        # Generate from scratch (all masked)
        tokens = sedd_model.sample(
            batch_size=n_per_class,
            class_labels=class_labels,
            num_steps=num_steps,
            temperature=1.0
        )
        
        all_tokens.append(tokens.cpu())
        all_labels.extend([class_idx] * n_per_class)
        log(f"  Class {class_idx} ({MODELNET40_CLASSES[class_idx]}): generated {n_per_class} samples")
    
    tokens = torch.cat(all_tokens, dim=0)
    labels = torch.tensor(all_labels)
    
    log(f"✓ SEDD generated {len(tokens)} tokens, shape: {tokens.shape}")
    return tokens, labels


@torch.no_grad()
def test_sedd_puncture_refill(sedd_model, val_tokens, val_labels, mask_interval=4, num_steps=20):
    """Test SEDD puncture and refill."""
    log(f"\nSEDD: Testing puncture refill (mask_interval={mask_interval})...")
    sedd_model.eval().to(DEVICE)
    
    # Take a small batch
    n_test = min(8, len(val_tokens))
    tokens = val_tokens[:n_test].to(DEVICE)
    labels = val_labels[:n_test].to(DEVICE)
    
    # Puncture (mask tokens)
    masked = tokens.clone()
    stride = mask_interval + 1
    for pos in range(SEQ_LEN):
        if pos % stride != 0:
            masked[:, pos] = MASK_ID
    
    pct_masked = (masked == MASK_ID).float().mean().item() * 100
    log(f"  Masked {pct_masked:.1f}% of tokens")
    
    # Refill using reverse diffusion
    num_timesteps = sedd_model.noise_schedule.num_timesteps
    stride_t = max(1, num_timesteps // num_steps)
    timesteps = list(range(num_timesteps - 1, -1, -stride_t))
    
    x = masked.clone()
    for t in timesteps:
        t_tensor = torch.full((n_test,), t, device=DEVICE, dtype=torch.long)
        logits = sedd_model(x, t_tensor, labels)
        
        mask = (x == MASK_ID)
        if not mask.any():
            break
        
        masked_logits = logits[mask]
        probs = F.softmax(masked_logits, dim=-1)
        samples = torch.multinomial(probs, 1).squeeze(-1)
        x_new = x.clone()
        x_new[mask] = samples
        x = x_new
    
    # Check recovery quality
    matches = (x == tokens).float().mean().item() * 100
    log(f"  Recovery accuracy: {matches:.1f}% tokens match original")
    
    return x.cpu(), labels.cpu()


@torch.no_grad()
def generate_dot_class_only(dot_model, n_per_class=2):
    """Generate tokens from class labels only using DoT."""
    log(f"\nDoT: Generating {n_per_class} samples per class from class labels only...")
    dot_model.eval().to(DEVICE)
    
    all_tokens = []
    all_labels = []
    
    for class_idx in range(min(5, NUM_CLASSES)):  # Test first 5 classes
        # Create prompt: [BOS, class_token]
        bos = torch.full((n_per_class, 1), BOS_TOKEN, device=DEVICE)
        cls_tokens = (BOS_TOKEN + class_idx + 1).unsqueeze(0).expand(n_per_class, 1).to(DEVICE)
        prompt = torch.cat([bos, cls_tokens], dim=1)  # [B, 2]
        
        # Generate
        generated = dot_model.generate(prompt, SEQ_LEN)  # [B, 2 + SEQ_LEN]
        tokens = generated[:, 2:]  # Strip BOS + class_token
        
        all_tokens.append(tokens.cpu())
        all_labels.extend([class_idx] * n_per_class)
        log(f"  Class {class_idx} ({MODELNET40_CLASSES[class_idx]}): generated {n_per_class} samples")
    
    tokens = torch.cat(all_tokens, dim=0)
    labels = torch.tensor(all_labels)
    
    log(f"✓ DoT generated {len(tokens)} tokens, shape: {tokens.shape}")
    return tokens, labels


@torch.no_grad()
def test_dot_prefix_completion(dot_model, val_tokens, val_labels, context_len=512):
    """Test DoT prefix completion."""
    log(f"\nDoT: Testing prefix completion (context_len={context_len})...")
    dot_model.eval().to(DEVICE)
    
    # Take a small batch
    n_test = min(8, len(val_tokens))
    tokens = val_tokens[:n_test].to(DEVICE)
    labels = val_labels[:n_test].to(DEVICE)
    
    # Create prompt: [BOS, class_token, prefix]
    bos = torch.full((n_test, 1), BOS_TOKEN, device=DEVICE)
    cls_tokens = (BOS_TOKEN + labels + 1).unsqueeze(1).to(DEVICE)
    prefix = tokens[:, :context_len]
    prompt = torch.cat([bos, cls_tokens, prefix], dim=1)  # [B, 2 + context_len]
    
    pct_kept = context_len / SEQ_LEN * 100
    log(f"  Keeping {pct_kept:.1f}% as prefix ({context_len} tokens)")
    
    # Complete the sequence
    rem = SEQ_LEN - context_len
    out = dot_model.generate(prompt, rem)  # [B, 2 + context_len + rem]
    completed = out[:, 2:]  # Strip BOS + class_token
    
    # Check prefix matches
    prefix_match = (completed[:, :context_len] == prefix).float().mean().item() * 100
    log(f"  Prefix preserved: {prefix_match:.1f}%")
    
    return completed.cpu(), labels.cpu()


def load_checkpoint(model_class, ckpt_path, model_type="sedd"):
    """Load model from checkpoint."""
    log(f"Loading {model_type} from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    model = model_class.load_from_checkpoint(ckpt_path, map_location=DEVICE)
    model.eval().to(DEVICE)
    log(f"✓ Loaded {model_type}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Test inference modes")
    parser.add_argument("--sedd_ckpt", type=str, default=None,
                        help="Path to SEDD checkpoint")
    parser.add_argument("--dot_ckpt", type=str, default=None,
                        help="Path to DoT checkpoint")
    parser.add_argument("--val_codes", type=str, default="../trash/data/val_codes.pt",
                        help="Path to validation codes")
    parser.add_argument("--out_dir", type=str, default="trash/test_inference",
                        help="Output directory")
    args = parser.parse_args()
    
    log("="*60)
    log("INFERENCE TEST: Class ID Generation & Puncture Refill")
    log("="*60)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load validation data for puncture tests
    log("\nLoading validation data...")
    val_data = torch.load(args.val_codes, map_location="cpu", weights_only=False)
    val_tokens = val_data.get("codes", val_data.get("tokens")).long()
    val_labels = val_data["labels"].long()
    log(f"Loaded {len(val_tokens)} validation samples")
    
    results = {}
    
    # Test SEDD
    if args.sedd_ckpt:
        try:
            sedd_model = load_checkpoint(DiscreteDiffusionTransformer, args.sedd_ckpt, "SEDD")
            
            # Test 1: Class ID only generation
            sedd_generated, sedd_gen_labels = generate_sedd_class_only(sedd_model, n_per_class=2)
            torch.save({"tokens": sedd_generated, "labels": sedd_gen_labels}, 
                      os.path.join(args.out_dir, "sedd_generated.pt"))
            
            # Test 2: Puncture refill
            sedd_recovered, sedd_rec_labels = test_sedd_puncture_refill(
                sedd_model, val_tokens, val_labels, mask_interval=4
            )
            torch.save({"tokens": sedd_recovered, "labels": sedd_rec_labels},
                      os.path.join(args.out_dir, "sedd_recovered.pt"))
            
            results["sedd_class_generation"] = True
            results["sedd_puncture_refill"] = True
        except Exception as e:
            log(f"SEDD tests failed: {e}", "ERROR")
            results["sedd_class_generation"] = False
            results["sedd_puncture_refill"] = False
    else:
        log("Skipping SEDD tests (no checkpoint provided)")
        results["sedd_class_generation"] = None
        results["sedd_puncture_refill"] = None
    
    # Test DoT
    if args.dot_ckpt:
        try:
            # Load DoT model
            sys.path.insert(0, os.path.join(_BASE, "models", "autoregressive"))
            from train_dot_mesh import NanoGpt
            
            dot_model = load_checkpoint(NanoGpt, args.dot_ckpt, "DoT")
            
            # Test 1: Class ID only generation
            dot_generated, dot_gen_labels = generate_dot_class_only(dot_model, n_per_class=2)
            torch.save({"tokens": dot_generated, "labels": dot_gen_labels},
                      os.path.join(args.out_dir, "dot_generated.pt"))
            
            # Test 2: Prefix completion
            dot_completed, dot_comp_labels = test_dot_prefix_completion(
                dot_model, val_tokens, val_labels, context_len=512
            )
            torch.save({"tokens": dot_completed, "labels": dot_comp_labels},
                      os.path.join(args.out_dir, "dot_completed.pt"))
            
            results["dot_class_generation"] = True
            results["dot_prefix_completion"] = True
        except Exception as e:
            log(f"DoT tests failed: {e}", "ERROR")
            results["dot_class_generation"] = False
            results["dot_prefix_completion"] = False
    else:
        log("Skipping DoT tests (no checkpoint provided)")
        results["dot_class_generation"] = None
        results["dot_prefix_completion"] = None
    
    # Summary
    log("\n" + "="*60)
    log("INFERENCE TEST SUMMARY")
    log("="*60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else ("✗ FAIL" if passed is False else "- SKIP")
        log(f"{test:30s}: {status}")
    
    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not None)
    log(f"\nTotal: {passed}/{total} tests passed")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "passed": passed,
        "total": total,
        "outputs": {
            "sedd_generated": "sedd_generated.pt",
            "sedd_recovered": "sedd_recovered.pt",
            "dot_generated": "dot_generated.pt",
            "dot_completed": "dot_completed.pt"
        }
    }
    
    with open(os.path.join(args.out_dir, "inference_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    log(f"\nOutput files in: {args.out_dir}/")
    log("- sedd_generated.pt: SEDD class-only generation")
    log("- sedd_recovered.pt: SEDD puncture refill")
    log("- dot_generated.pt: DoT class-only generation")
    log("- dot_completed.pt: DoT prefix completion")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
