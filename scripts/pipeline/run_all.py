#!/usr/bin/env python3
"""
run_all.py
----------
Complete pipeline: Train all models → Run semantic channel evaluation.

This is the ONE script you run for the full workflow.

Usage:
    # Full pipeline: Train + Evaluate
    python run_all.py

    # Skip training, only run evaluation (requires trained models)
    python run_all.py --eval_only

    # Use custom paths
    python run_all.py --data_dir my_data --out_dir my_results

    # Skip specific models
    python run_all.py --skip_dot  # Train classifier + SEDD only
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_TRASH = os.path.join(os.path.dirname(_BASE), "trash")


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] [{level}] {msg}")


def run_command(cmd, desc):
    """Run a command and stream output."""
    log(f"Running: {desc}")
    log(f"Command: {cmd}")
    
    process = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    
    for line in process.stdout:
        print(line, end="")
    
    return_code = process.wait()
    
    if return_code == 0:
        log(f"✓ {desc} completed")
        return True
    else:
        log(f"✗ {desc} failed (code {return_code})", "ERROR")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Train models + Run semantic channel evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (train + evaluate)
  python run_all.py

  # Evaluation only (requires trained models)
  python run_all.py --eval_only

  # Custom paths
  python run_all.py --data_dir my_data --out_dir my_results

  # Train only classifier, then evaluate
  python run_all.py --skip_sedd --skip_dot
        """
    )
    
    # Data paths
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(_TRASH, "data"),
                        help="Path to train_codes.pt and val_codes.pt")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_TRASH, "pipeline_results"),
                        help="Output directory")
    
    # Training control
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only run evaluation")
    parser.add_argument("--skip_classifier", action="store_true")
    parser.add_argument("--skip_sedd", action="store_true")
    parser.add_argument("--skip_dot", action="store_true")
    
    # Training config
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--sedd_epochs", type=int, default=200)
    parser.add_argument("--dot_epochs", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=1)
    
    # Evaluation config
    parser.add_argument("--sedd_steps", type=int, default=100,
                        help="SEDD diffusion steps for generation")
    
    args = parser.parse_args()
    
    print("="*70)
    print("  COMPLETE PIPELINE")
    print("="*70)
    print(f"Data:   {args.data_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Mode:   {'EVAL ONLY' if args.eval_only else 'TRAIN + EVAL'}")
    print("="*70)
    
    os.makedirs(args.out_dir, exist_ok=True)
    training_dir = os.path.join(args.out_dir, "training")
    
    results = {"training": {}, "evaluation": {}}
    
    # ── PHASE 1: TRAINING ───────────────────────────────────────────────────
    if not args.eval_only:
        log("PHASE 1: TRAINING MODELS")
        
        train_script = os.path.join(_BASE, "scripts", "training", "train_all.py")
        
        skip_flags = []
        if args.skip_classifier:
            skip_flags.append("--skip_classifier")
        if args.skip_sedd:
            skip_flags.append("--skip_sedd")
        if args.skip_dot:
            skip_flags.append("--skip_dot")
        
        cmd = f"""python {train_script} \
            --data_dir {args.data_dir} \
            --out_dir {training_dir} \
            --gpus {args.gpus} \
            {' '.join(skip_flags)}"""
        
        training_success = run_command(cmd, "Training")
        results["training"]["success"] = training_success
        
        if not training_success:
            log("Training failed, stopping pipeline", "ERROR")
            return 1
    else:
        log("PHASE 1: SKIPPED (--eval_only)")
        results["training"]["skipped"] = True
    
    # ── PHASE 2: EVALUATION ──────────────────────────────────────────────────
    log("PHASE 2: SEMANTIC CHANNEL EVALUATION")
    
    eval_script = os.path.join(_BASE, "evaluation", "semantic_channel", 
                               "mesh_transmit_semantic_channel.py")
    
    # Determine checkpoint paths
    classifier_ckpt = os.path.join(training_dir, "classifier", "checkpoints")
    sedd_ckpt = os.path.join(training_dir, "sedd")
    dot_ckpt = os.path.join(training_dir, "dot")
    
    if args.eval_only:
        # If eval_only, use default locations or let user specify
        if not os.path.exists(classifier_ckpt):
            classifier_ckpt = os.path.join(_TRASH, "training_runs", "classifier", "checkpoints")
        if not os.path.exists(sedd_ckpt):
            sedd_ckpt = os.path.join(_TRASH, "training_runs", "sedd")
        if not os.path.exists(dot_ckpt):
            dot_ckpt = os.path.join(_TRASH, "training_runs", "dot")
    
    val_codes = os.path.join(args.data_dir, "val_codes.pt")
    eval_out = os.path.join(args.out_dir, "semantic_channel_results")
    
    skip_flags = []
    if args.skip_sedd:
        skip_flags.append("--skip_sedd_full_gen")
    if args.skip_dot:
        skip_flags.append("--skip_dot_full_gen")
    
    cmd = f"""python {eval_script} \
        --classifier_ckpt {classifier_ckpt} \
        --sedd_ckpt {sedd_ckpt} \
        --dot_ckpt {dot_ckpt} \
        --real_codes_path {val_codes} \
        --out_dir {eval_out} \
        --sedd_steps {args.sedd_steps} \
        {' '.join(skip_flags)}"""
    
    eval_success = run_command(cmd, "Semantic Channel Evaluation")
    results["evaluation"]["success"] = eval_success
    
    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    
    # Save report
    report_path = os.path.join(args.out_dir, "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "results": results,
            "paths": {
                "training_dir": training_dir,
                "eval_dir": eval_out,
                "plots": os.path.join(eval_out, "plots")
            }
        }, f, indent=2)
    
    log(f"Report saved: {report_path}")
    log(f"Plots: {os.path.join(eval_out, 'plots')}")
    
    if all(v.get("success") for v in results.values() if isinstance(v, dict)):
        log("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        return 0
    else:
        log("✗ SOME STEPS FAILED", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
