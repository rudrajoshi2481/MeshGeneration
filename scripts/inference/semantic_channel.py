#!/usr/bin/env python3
"""
semantic_channel.py
---------------------
Run semantic channel evaluation on trained models.

This evaluates:
  - Clean accuracy (real codes)
  - SEDD full generation accuracy
  - DoT full generation accuracy
  - SEDD puncture+refill accuracy vs % missing
  - DoT prefix completion accuracy vs % missing

Usage:
    # Basic usage (provide all checkpoints)
    python semantic_channel.py \
        --classifier_ckpt trash/training_runs/classifier/checkpoints/best.ckpt \
        --sedd_ckpt trash/training_runs/sedd/checkpoints/best.ckpt \
        --dot_ckpt trash/training_runs/dot/checkpoints/dot_final.ckpt

    # Use defaults (looks for models in trash/training_runs/)
    python semantic_channel.py

    # Custom output directory
    python semantic_channel.py --out_dir my_results/
"""

import os
import sys
import argparse

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_TRASH = os.path.join(os.path.dirname(_BASE), "trash")

# Import the actual implementation
sys.path.insert(0, os.path.join(_BASE, "evaluation", "semantic_channel"))
from mesh_transmit_semantic_channel import main as _original_main


def main():
    parser = argparse.ArgumentParser(
        description="Semantic channel evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use trained models from default location
  python semantic_channel.py

  # Specify custom checkpoints
  python semantic_channel.py \\
      --classifier_ckpt path/to/classifier.ckpt \\
      --sedd_ckpt path/to/sedd.ckpt \\
      --dot_ckpt path/to/dot.ckpt

  # Skip some evaluations
  python semantic_channel.py --skip_sedd_full_gen --skip_dot_full_gen
        """
    )
    
    # Data paths
    parser.add_argument("--classifier_ckpt", type=str,
                        default=os.path.join(_TRASH, "training_runs", "classifier", 
                                             "checkpoints", "classifier_conditional-epoch=49-val_acc=0.5702.ckpt"),
                        help="Path to trained classifier .ckpt")
    parser.add_argument("--sedd_ckpt", type=str,
                        default=os.path.join(_TRASH, "training_runs", "sedd"),
                        help="Path to trained SEDD checkpoint (file or dir)")
    parser.add_argument("--dot_ckpt", type=str,
                        default=os.path.join(_TRASH, "training_runs", "dot"),
                        help="Path to trained DoT checkpoint (file or dir)")
    parser.add_argument("--real_codes_path", type=str,
                        default=os.path.join(_TRASH, "data", "val_codes.pt"),
                        help="Path to validation codes .pt file")
    
    # Output
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_TRASH, "semantic_channel_results"),
                        help="Output directory for results and plots")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_per_class", type=int, default=5,
                        help="Samples per class for full generation")
    parser.add_argument("--sedd_n_eval", type=int, default=200,
                        help="Number of samples for SEDD puncture eval")
    parser.add_argument("--dot_n_eval", type=int, default=200,
                        help="Number of samples for DoT prefix eval")
    parser.add_argument("--sedd_steps", type=int, default=100,
                        help="SEDD diffusion steps")
    
    # Skips
    parser.add_argument("--skip_sedd_full_gen", action="store_true")
    parser.add_argument("--skip_dot_full_gen", action="store_true")
    
    args = parser.parse_args()
    
    print("="*70)
    print("  SEMANTIC CHANNEL EVALUATION")
    print("="*70)
    print(f"Classifier: {args.classifier_ckpt}")
    print(f"SEDD:       {args.sedd_ckpt}")
    print(f"DoT:        {args.dot_ckpt}")
    print(f"Output:     {args.out_dir}")
    print("="*70)
    
    # Auto-find checkpoints if directories provided
    if os.path.isdir(args.sedd_ckpt):
        args.sedd_ckpt = _find_checkpoint(args.sedd_ckpt, "sedd")
    if os.path.isdir(args.dot_ckpt):
        args.dot_ckpt = _find_checkpoint(args.dot_ckpt, "dot")
    
    # Run the actual evaluation
    sys.argv = [sys.argv[0]] + sys.argv[1:]  # Preserve args
    return _original_main()


def _find_checkpoint(directory, model_type):
    """Find best checkpoint in directory."""
    import glob
    
    patterns = {
        "sedd": ["*best*.ckpt", "*epoch*.ckpt"],
        "dot": ["*dot_final*.pt", "*epoch*.ckpt"],
    }
    
    for pattern in patterns.get(model_type, []):
        matches = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
        if matches:
            return matches[0]
    
    return None


if __name__ == "__main__":
    sys.exit(main())
