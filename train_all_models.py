"""
train_all_models.py
-------------------
Train all models with full configuration (not test mode).

Usage:
    python train_all_models.py --data_dir ../trash/data --out_dir trash/training_runs
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))

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
    },
    "dot": {
        "epochs": 100,
        "batch_size": 16,
        "mode": "small",
        "n_train": -1,
    }
}


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_training(name, cmd, cwd, log_file):
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
    
    # Run command and stream output to both console and log file
    import subprocess
    process = subprocess.Popen(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    
    # Stream output in real-time
    with open(log_file, "a") as f:
        for line in process.stdout:
            print(line, end="")  # Print to console
            f.write(line)        # Write to log file
            f.flush()            # Flush immediately
    
    return_code = process.wait()
    
    # Append final status to log
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
    parser.add_argument("--data_dir", type=str, default="../trash/data",
                        help="Path to train_codes.pt and val_codes.pt")
    parser.add_argument("--out_dir", type=str, default="trash/training_runs",
                        help="Output directory")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--skip_classifier", action="store_true")
    parser.add_argument("--skip_sedd", action="store_true")
    parser.add_argument("--skip_dot", action="store_true")
    args = parser.parse_args()
    
    # Resolve absolute paths
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
        
        cmd = f"""cd {os.path.join(_BASE, "models", "classifier")} && \
python train_classifier.py \
    --tokens_path {os.path.join(data_dir, "train_codes.pt")} \
    --mode {TRAIN_CONFIG["classifier"]["mode"]} \
    --out_dir {out} \
    --epochs {TRAIN_CONFIG["classifier"]["epochs"]} \
    --batch_size {TRAIN_CONFIG["classifier"]["batch_size"]} \
    --lr {TRAIN_CONFIG["classifier"]["lr"]} \
    --gpus {args.gpus}"""
        
        results["classifier"] = run_training("Classifier", cmd, None, log_file)
    else:
        log("Skipping Classifier")
        results["classifier"] = None
    
    # 2. Train SEDD
    if not args.skip_sedd:
        out = os.path.join(out_dir, "sedd")
        log_file = os.path.join(out, "training.log")
        
        n_train = TRAIN_CONFIG["sedd"]["n_train"]
        n_train_arg = f"--n_train {n_train}" if n_train > 0 else "--n_train -1"
        
        cmd = f"""cd {os.path.join(_BASE, "models", "diffusion")} && \
python train_sedd.py \
    --mode {TRAIN_CONFIG["sedd"]["mode"]} \
    --epochs {TRAIN_CONFIG["sedd"]["epochs"]} \
    --batch_size {TRAIN_CONFIG["sedd"]["batch_size"]} \
    {n_train_arg} \
    --gpus {args.gpus}"""
        
        results["sedd"] = run_training("SEDD", cmd, None, log_file)
    else:
        log("Skipping SEDD")
        results["sedd"] = None
    
    # 3. Train DoT
    if not args.skip_dot:
        out = os.path.join(out_dir, "dot")
        log_file = os.path.join(out, "training.log")
        
        n_train = TRAIN_CONFIG["dot"]["n_train"]
        n_train_arg = f"--n_train {n_train}" if n_train > 0 else "--n_train -1"
        
        cmd = f"""cd {os.path.join(_BASE, "models", "autoregressive")} && \
python train_dot_mesh.py \
    --mode {TRAIN_CONFIG["dot"]["mode"]} \
    --epochs {TRAIN_CONFIG["dot"]["epochs"]} \
    --batch_size {TRAIN_CONFIG["dot"]["batch_size"]} \
    {n_train_arg} \
    --gpus {args.gpus}"""
        
        results["dot"] = run_training("DoT", cmd, None, log_file)
    else:
        log("Skipping DoT")
        results["dot"] = None
    
    # Summary
    log("\n" + "="*60)
    log("TRAINING SUMMARY")
    log("="*60)
    
    for model, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        log(f"{model:15s}: {status}")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": TRAIN_CONFIG,
        "results": results,
        "output_dir": out_dir,
    }
    
    with open(os.path.join(out_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    log(f"\nReport saved: {os.path.join(out_dir, 'training_report.json')}")
    log("Check individual training.log files in each subdirectory")
    
    # Return success if all completed
    completed = [v for v in results.values() if v is not None]
    passed = sum(1 for v in completed if v)
    
    if passed == len(completed) and len(completed) > 0:
        log("\n✓ ALL MODELS TRAINED SUCCESSFULLY!")
        return 0
    else:
        log(f"\n✗ SOME MODELS FAILED ({passed}/{len(completed)})", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
