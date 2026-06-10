#!/usr/bin/env python3
"""
run_all.py
----------
Unified script that runs the ENTIRE semantic channel pipeline end-to-end:
  1. Extract codebook indices from MeshVQVAE (if not already done)
  2. Train classifier on real codes
  3. Train SEDD diffusion model
  4. Train DoT (NanoGPT) autoregressive model
  5. Run semantic channel evaluation (puncture + recovery + classification)

All outputs are saved to the trash/ directory.

Usage:
  python run_all.py                          # run everything
  python run_all.py --skip_training          # skip training, run eval only
  python run_all.py --eval_only              # same as --skip_training
  python run_all.py --sedd_steps 20          # override SEDD diffusion steps
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))      # evaluation/semantic_channel/
_BASE     = os.path.dirname(os.path.dirname(_HERE))           # MeshGeneration/
_SEMENTIC = os.path.dirname(_BASE)                            # sementic_channel_project/
_TRASH    = os.path.join(_SEMENTIC, "trash")
DATA_DIR  = os.path.join(_TRASH, "data")

# Sub-directories
CLASSIFIER  = os.path.join(_BASE, "models", "classifier")
AUTOREGRESSIVE = os.path.join(_BASE, "models", "autoregressive")
DIFFUSION = os.path.join(_BASE, "diffusion_model")
SEMCHAN   = _HERE  # Already in evaluation/semantic_channel/


def run_cmd(cmd: list, cwd: str, desc: str, env=None):
    """Run a command and stream output, raising on failure."""
    print(f"\n{'─'*60}")
    print(f"  [{desc}]")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  cwd: {cwd}")
    print(f"{'─'*60}\n")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, env=merged_env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        raise RuntimeError(f"Step '{desc}' failed with exit code {result.returncode}")
    print(f"  ✓ Done in {elapsed:.1f}s")
    return result


def find_best_ckpt(ckpt_dir: str, prefix: str = "", metric_key: str = "val_loss",
                   mode: str = "min") -> str:
    """Find the best checkpoint in a directory by parsing filename metrics."""
    if not os.path.isdir(ckpt_dir):
        return ""
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and prefix in f]
    if not ckpts:
        return ""
    
    best_path = ""
    best_val = float("inf") if mode == "min" else float("-inf")
    
    for ckpt in ckpts:
        # Parse metric from filename like "sedd-epoch=0041-val_loss=1.5629.ckpt"
        parts = ckpt.replace(".ckpt", "").split("-")
        for part in parts:
            if metric_key.replace("_", "_") in part:
                try:
                    val = float(part.split("=")[1])
                    if (mode == "min" and val < best_val) or (mode == "max" and val > best_val):
                        best_val = val
                        best_path = os.path.join(ckpt_dir, ckpt)
                except (ValueError, IndexError):
                    continue
    
    return best_path


def find_latest_run(base_dir: str, prefix: str = "") -> str:
    """Find the latest run directory."""
    if not os.path.isdir(base_dir):
        return ""
    dirs = sorted([d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)],
                  reverse=True)
    return os.path.join(base_dir, dirs[0]) if dirs else ""


def main():
    parser = argparse.ArgumentParser(description="Run full semantic channel pipeline")
    
    # Training options
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip all training steps, use existing checkpoints")
    parser.add_argument("--eval_only", action="store_true",
                        help="Alias for --skip_training")
    
    # Classifier options
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    
    # SEDD options
    parser.add_argument("--sedd_mode", type=str, default="small",
                        choices=["small", "medium", "full"])
    parser.add_argument("--sedd_epochs", type=int, default=200)
    parser.add_argument("--sedd_batch_size", type=int, default=16)
    
    # DoT options
    parser.add_argument("--dot_mode", type=str, default="small",
                        choices=["small", "full"])
    parser.add_argument("--dot_epochs", type=int, default=100)
    parser.add_argument("--dot_batch_size", type=int, default=8)
    
    # Eval options
    parser.add_argument("--sedd_steps", type=int, default=20,
                        help="SEDD diffusion steps for evaluation")
    parser.add_argument("--n_per_class", type=int, default=10,
                        help="Samples per class for full-gen eval")
    parser.add_argument("--dot_n_eval", type=int, default=200,
                        help="Samples for DoT prefix eval")
    parser.add_argument("--skip_dot_full_gen", action="store_true",
                        help="Skip slow DoT full-generation step")
    
    # Existing checkpoints (override auto-detection)
    parser.add_argument("--classifier_ckpt", type=str, default=None)
    parser.add_argument("--sedd_ckpt", type=str, default=None)
    parser.add_argument("--dot_ckpt", type=str, default=None)
    
    args = parser.parse_args()
    if args.eval_only:
        args.skip_training = True

    print(f"\n{'='*65}")
    print(f"  Semantic Channel Pipeline — Full Run")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Output: {_TRASH}")
    print(f"  Training: {'SKIP' if args.skip_training else 'ENABLED'}")
    print(f"{'='*65}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 0: Verify data exists
    # ══════════════════════════════════════════════════════════════════════════
    train_codes = os.path.join(DATA_DIR, "train_codes.pt")
    val_codes = os.path.join(DATA_DIR, "val_codes.pt")
    
    if not os.path.exists(train_codes) or not os.path.exists(val_codes):
        print(f"[ERROR] Data files not found:")
        print(f"  Expected: {train_codes}")
        print(f"  Expected: {val_codes}")
        print(f"  Please run extract_codes.py first to generate codebook indices.")
        sys.exit(1)
    
    print(f"[✓] Data files found: {train_codes}, {val_codes}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Train Classifier
    # ══════════════════════════════════════════════════════════════════════════
    classifier_out = os.path.join(_TRASH, "classifier_real")
    classifier_ckpt = args.classifier_ckpt

    if not args.skip_training and classifier_ckpt is None:
        print("\n[STEP 1] Training classifier on real codes ...")
        run_cmd([
            sys.executable, "train_classifier.py",
            "--tokens_path", train_codes,
            "--mode", "conditional",
            "--out_dir", classifier_out,
            "--epochs", str(args.classifier_epochs),
            "--lr", str(args.classifier_lr),
            "--gpus", "1",
            "--batch_size", "64",
        ], cwd=CLASSIFIER, desc="Classifier Training")
    
    # Find best classifier checkpoint
    if classifier_ckpt is None:
        classifier_ckpt = find_best_ckpt(
            os.path.join(classifier_out, "checkpoints"),
            prefix="classifier", metric_key="val_acc", mode="max"
        )
    
    if not classifier_ckpt or not os.path.exists(classifier_ckpt):
        print(f"[ERROR] No classifier checkpoint found in {classifier_out}/checkpoints/")
        sys.exit(1)
    print(f"[✓] Classifier ckpt: {classifier_ckpt}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Train SEDD
    # ══════════════════════════════════════════════════════════════════════════
    sedd_ckpt = args.sedd_ckpt

    if not args.skip_training and sedd_ckpt is None:
        print("\n[STEP 2] Training SEDD diffusion model ...")
        run_cmd([
            sys.executable, "train_sedd.py",
            "--mode", args.sedd_mode,
            "--gpus", "1",
            "--batch_size", str(args.sedd_batch_size),
            "--n_train", "-1",
            "--epochs", str(args.sedd_epochs),
        ], cwd=DIFFUSION, desc="SEDD Training")
    
    # Find best SEDD checkpoint
    if sedd_ckpt is None:
        sedd_base = os.path.join(_TRASH, "sedd_runs")
        latest_sedd = find_latest_run(sedd_base, prefix="sedd_")
        if latest_sedd:
            sedd_ckpt = find_best_ckpt(
                os.path.join(latest_sedd, "checkpoints"),
                prefix="sedd", metric_key="val_loss", mode="min"
            )
    
    if not sedd_ckpt or not os.path.exists(sedd_ckpt):
        print(f"[ERROR] No SEDD checkpoint found")
        sys.exit(1)
    print(f"[✓] SEDD ckpt: {sedd_ckpt}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Train DoT (NanoGPT)
    # ══════════════════════════════════════════════════════════════════════════
    dot_ckpt = args.dot_ckpt

    if not args.skip_training and dot_ckpt is None:
        print("\n[STEP 3] Training DoT (NanoGPT) ...")
        run_cmd([
            sys.executable, "train_dot_mesh.py",
            "--mode", args.dot_mode,
            "--n_train", "-1",
            "--epochs", str(args.dot_epochs),
            "--batch_size", str(args.dot_batch_size),
        ], cwd=AUTOREGRESSIVE, desc="DoT Training")
    
    # Find best DoT checkpoint
    if dot_ckpt is None:
        dot_base = os.path.join(_TRASH, "dot_runs")
        latest_dot = find_latest_run(dot_base, prefix="dot_")
        if latest_dot:
            dot_ckpt = find_best_ckpt(
                os.path.join(latest_dot, "checkpoints"),
                prefix="dot", metric_key="val_loss", mode="min"
            )
    
    if not dot_ckpt or not os.path.exists(dot_ckpt):
        print(f"[WARN] No DoT checkpoint found — DoT eval will be skipped")
        dot_ckpt = None

    if dot_ckpt:
        print(f"[✓] DoT ckpt: {dot_ckpt}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Run Semantic Channel Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[STEP 4] Running semantic channel evaluation ...")
    eval_cmd = [
        sys.executable, "-u", "mesh_transmit_semantic_channel.py",
        "--classifier_ckpt", classifier_ckpt,
        "--sedd_ckpt", sedd_ckpt,
        "--real_codes_path", val_codes,
        "--n_per_class", str(args.n_per_class),
        "--sedd_steps", str(args.sedd_steps),
        "--dot_n_eval", str(args.dot_n_eval),
    ]
    if dot_ckpt:
        eval_cmd += ["--dot_ckpt", dot_ckpt]
    if args.skip_dot_full_gen:
        eval_cmd += ["--skip_dot_full_gen"]
    
    env = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    run_cmd(eval_cmd, cwd=SEMCHAN, desc="Semantic Channel Evaluation", env=env)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Print final results
    # ══════════════════════════════════════════════════════════════════════════
    results_path = os.path.join(_TRASH, "semantic_channel_results", "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        
        print(f"\n{'═'*65}")
        print(f"  FINAL RESULTS")
        print(f"{'═'*65}")
        print(f"  Clean accuracy:           {results.get('clean_accuracy', 0):.2f}%")
        if "sedd_full_accuracy" in results:
            print(f"  SEDD full-gen accuracy:   {results['sedd_full_accuracy']:.2f}%")
        if "dot_full_accuracy" in results:
            print(f"  DoT full-gen accuracy:    {results['dot_full_accuracy']:.2f}%")
        if "sedd_partial" in results:
            print(f"\n  SEDD Puncture Recovery:")
            for r in results["sedd_partial"]:
                print(f"    {r['pct_missing']:5.1f}% missing → {r['accuracy']:.2f}% accuracy")
        if "dot_partial" in results:
            print(f"\n  DoT Prefix Completion:")
            for r in results["dot_partial"]:
                print(f"    {r['pct_missing']:5.1f}% missing → {r['accuracy']:.2f}% accuracy")
        print(f"\n  Results: {results_path}")
        print(f"  Plot:    {os.path.join(_TRASH, 'semantic_channel_results', 'plots', 'accuracy_vs_missing_tokens.png')}")
        print(f"{'═'*65}\n")
    else:
        print(f"[WARN] Results file not found at {results_path}")

    print("[✓] Pipeline complete!")


if __name__ == "__main__":
    main()
