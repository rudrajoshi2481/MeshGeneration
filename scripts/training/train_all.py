#!/usr/bin/env python3
"""
train_all.py
------------
Train all models with full configuration.

Runs sequentially: Classifier → SEDD → DoT
(GPU can only handle one at a time)

Usage:
    python train_all.py --data_dir trash/data --out_dir trash/training_runs
    python train_all.py --skip_classifier  # Train only SEDD + DoT
    python train_all.py --skip_sedd --skip_dot  # Train only classifier
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── full training configuration ───────────────────────────────────────────────
TRAIN_CONFIG = {
    "classifier": {
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
        "mode": "conditional",
    },
    "sedd": {
        "epochs": 200,
        "batch_size": 16,
        "mode": "small",  # small, medium, full
        "n_train": -1,  # -1 = all data
        "plot_every": 5,
    },
    "dot": {
        "epochs": 100,
        "batch_size": 16,
        "mode": "small",
        "n_train": -1,
        "plot_every": 5,
    }
}


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_training(name, cmd, log_file):
    """Run training command and stream output to log file in real-time."""
    log(f"\n{'='*60}")
    log(f"Starting {name} Training")
    log(f"{'='*60}")
    log(f"Command: {cmd}")
    log(f"Log: {log_file}")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Write header to log file
    with open(log_file, "w") as f:
        f.write(f"Training Log: {name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"{'='*80}\n\n")
    
    # Run command and stream output
    process = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    
    with open(log_file, "a") as f:
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
    
    return_code = process.wait()
    
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Return Code: {return_code}\n")
    
    if return_code == 0:
        log(f"✓ {name} training completed successfully")
    else:
        log(f"✗ {name} training failed (code {return_code})", "ERROR")
    
    return return_code == 0


def main():
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--data_dir", type=str, default="trash/data",
                        help="Path to train_codes.pt and val_codes.pt")
    parser.add_argument("--out_dir", type=str, default="trash/training_runs",
                        help="Output directory")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--skip_classifier", action="store_true")
    parser.add_argument("--skip_sedd", action="store_true")
    parser.add_argument("--skip_dot", action="store_true")
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    
    log("="*60)
    log("FULL MODEL TRAINING")
    log("="*60)
    log(f"Data: {data_dir}")
    log(f"Output: {out_dir}")
    log(f"GPUs: {args.gpus}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    # 1. Train Classifier
    if not args.skip_classifier:
        out = os.path.join(out_dir, "classifier")
        log_file = os.path.join(out, "training.log")
        
        cfg = TRAIN_CONFIG["classifier"]
        cmd = f"""cd {os.path.join(_BASE, "models", "classifier")} && \
python train_classifier.py \
    --tokens_path {os.path.join(data_dir, "train_codes.pt")} \
    --mode {cfg["mode"]} \
    --out_dir {out} \
    --epochs {cfg["epochs"]} \
    --batch_size {cfg["batch_size"]} \
    --lr {cfg["lr"]} \
    --gpus {args.gpus}"""
        
        results["classifier"] = run_training("Classifier", cmd, log_file)
    
    # 2. Train SEDD
    if not args.skip_sedd:
        out = os.path.join(out_dir, "sedd")
        log_file = os.path.join(out, "training.log")
        
        cfg = TRAIN_CONFIG["sedd"]
        n_train_arg = "--n_train -1" if cfg["n_train"] < 0 else f"--n_train {cfg['n_train']}"
        
        cmd = f"""cd {os.path.join(_BASE, "models", "diffusion")} && \
python train_sedd.py \
    --mode {cfg["mode"]} \
    --epochs {cfg["epochs"]} \
    --batch_size {cfg["batch_size"]} \
    {n_train_arg} \
    --plot_every {cfg["plot_every"]} \
    --gpus {args.gpus}"""
        
        results["sedd"] = run_training("SEDD", cmd, log_file)
    
    # 3. Train DoT
    if not args.skip_dot:
        out = os.path.join(out_dir, "dot")
        log_file = os.path.join(out, "training.log")
        
        cfg = TRAIN_CONFIG["dot"]
        n_train_arg = "--n_train -1" if cfg["n_train"] < 0 else f"--n_train {cfg['n_train']}"
        
        cmd = f"""cd {os.path.join(_BASE, "models", "autoregressive")} && \
python train_dot_mesh.py \
    --mode {cfg["mode"]} \
    --data_dir {data_dir} \
    --out_base {out} \
    --epochs {cfg["epochs"]} \
    --batch_size {cfg["batch_size"]} \
    {n_train_arg} \
    --plot_every {cfg["plot_every"]} \
    --gpus {args.gpus}"""
        
        results["dot"] = run_training("DoT", cmd, log_file)
    
    # Summary
    log("\n" + "="*60)
    log("TRAINING SUMMARY")
    log("="*60)
    
    for model, passed in results.items():
        status = "✓ SUCCESS" if passed else "✗ FAILED"
        log(f"{model:15s}: {status}")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": TRAIN_CONFIG,
        "results": results,
        "output_dir": out_dir,
    }
    
    with open(os.path.join(out_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    log(f"\nReport saved: {os.path.join(out_dir, 'training_report.json')}")
    
    passed = sum(1 for v in results.values() if v)
    if passed == len(results) and len(results) > 0:
        log("\n✓ ALL MODELS TRAINED SUCCESSFULLY!")
        return 0
    else:
        log(f"\n✗ SOME MODELS FAILED ({passed}/{len(results)})", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
