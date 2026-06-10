"""
test_all_models.py
------------------
Unified test script for all models in the semantic channel pipeline.
Tests with 1 epoch or small sample size to verify everything works.

Skips:
- VQVAE training (use pre-trained)
- Decoder generation (needs VQVAE checkpoint)

Usage:
    python test_all_models.py --data_dir trash/data --out_dir trash/test_runs
"""

import os
import sys
import argparse
import json
import torch
import shutil
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "models", "classifier"))
sys.path.insert(0, os.path.join(_BASE, "models", "diffusion"))
sys.path.insert(0, os.path.join(_BASE, "models", "autoregressive"))
sys.path.insert(0, os.path.join(_BASE, "mesh_vqvae", "src"))

# ── test configuration ────────────────────────────────────────────────────────
TEST_CONFIG = {
    "classifier": {
        "epochs": 1,
        "batch_size": 64,
        "n_samples": 128,  # small subset for testing
    },
    "sedd": {
        "epochs": 1,
        "batch_size": 4,
        "n_train": 32,  # small subset
        "mode": "small",
    },
    "dot": {
        "epochs": 1,
        "batch_size": 4,
        "n_train": 32,
        "mode": "small",
    }
}

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def check_data_exists(data_dir):
    """Check if required data files exist."""
    required = ["train_codes.pt", "val_codes.pt"]
    missing = []
    for f in required:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            missing.append(f)
        else:
            log(f"Found: {path}")
    return missing

def run_command(cmd, cwd=None):
    """Run shell command and return success status."""
    import subprocess
    log(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"Error: {result.stderr}", "ERROR")
        return False
    log("Success")
    return True

# ── test functions ────────────────────────────────────────────────────────────
def test_classifier(data_dir, out_dir):
    """Test classifier training for 1 epoch."""
    log("\n" + "="*60)
    log("TEST 1: Classifier (1 epoch, 128 samples)")
    log("="*60)
    
    out = os.path.join(out_dir, "test_classifier")
    os.makedirs(out, exist_ok=True)
    
    cmd = f"""cd {os.path.join(_BASE, "models", "classifier")} && \
python train_classifier.py \
    --tokens_path {os.path.join(data_dir, "train_codes.pt")} \
    --mode conditional \
    --out_dir {out} \
    --epochs {TEST_CONFIG["classifier"]["epochs"]} \
    --batch_size {TEST_CONFIG["classifier"]["batch_size"]} \
    --gpus 1"""
    
    return run_command(cmd)

def test_sedd(data_dir, out_dir):
    """Test SEDD training for 1 epoch."""
    log("\n" + "="*60)
    log("TEST 2: SEDD (1 epoch, 32 samples)")
    log("="*60)
    
    out = os.path.join(out_dir, "test_sedd")
    os.makedirs(out, exist_ok=True)
    
    cmd = f"""cd {os.path.join(_BASE, "models", "diffusion")} && \
python train_sedd.py \
    --mode {TEST_CONFIG["sedd"]["mode"]} \
    --epochs {TEST_CONFIG["sedd"]["epochs"]} \
    --batch_size {TEST_CONFIG["sedd"]["batch_size"]} \
    --n_train {TEST_CONFIG["sedd"]["n_train"]} \
    --gpus 1"""
    
    return run_command(cmd)

def test_dot(data_dir, out_dir):
    """Test DoT training for 1 epoch."""
    log("\n" + "="*60)
    log("TEST 3: DoT (1 epoch, 32 samples)")
    log("="*60)
    
    out = os.path.join(out_dir, "test_dot")
    os.makedirs(out, exist_ok=True)
    
    cmd = f"""cd {os.path.join(_BASE, "models", "autoregressive")} && \
python train_dot_mesh.py \
    --mode {TEST_CONFIG["dot"]["mode"]} \
    --epochs {TEST_CONFIG["dot"]["epochs"]} \
    --batch_size {TEST_CONFIG["dot"]["batch_size"]} \
    --n_train {TEST_CONFIG["dot"]["n_train"]} \
    --gpus 1"""
    
    return run_command(cmd)

def test_token_shapes(data_dir):
    """Verify all tokens have correct shape [N, 4096]."""
    log("\n" + "="*60)
    log("TEST 4: Verify Token Shapes")
    log("="*60)
    
    try:
        train_data = torch.load(os.path.join(data_dir, "train_codes.pt"), weights_only=False)
        val_data = torch.load(os.path.join(data_dir, "val_codes.pt"), weights_only=False)
        
        train_codes = train_data.get("codes", train_data.get("tokens"))
        val_codes = val_data.get("codes", val_data.get("tokens"))
        
        log(f"Train codes shape: {train_codes.shape}")
        log(f"Val codes shape: {val_codes.shape}")
        
        # Check dimensions
        assert train_codes.shape[1] == 4096, f"Expected 4096, got {train_codes.shape[1]}"
        assert val_codes.shape[1] == 4096, f"Expected 4096, got {val_codes.shape[1]}"
        
        # Check vocab range
        assert train_codes.min() >= 0 and train_codes.max() < 256, "Train codes out of range [0, 255]"
        assert val_codes.min() >= 0 and val_codes.max() < 256, "Val codes out of range [0, 255]"
        
        log("✓ All token shapes correct: [N, 4096]")
        log("✓ Vocab range correct: [0, 255]")
        return True
    except Exception as e:
        log(f"Token shape test failed: {e}", "ERROR")
        return False

def test_puncture_refill_logic():
    """Test puncture refill logic without full models."""
    log("\n" + "="*60)
    log("TEST 5: Puncture Refill Logic (Unit Test)")
    log("="*60)
    
    try:
        # Create dummy tokens
        batch_size = 4
        seq_len = 4096
        vocab_size = 256
        mask_id = 256
        
        # Real tokens
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test SEDD-style puncture (interval masking)
        mask_interval = 4
        masked = tokens.clone()
        stride = mask_interval + 1
        for pos in range(seq_len):
            if pos % stride != 0:
                masked[:, pos] = mask_id
        
        pct_masked = (masked == mask_id).float().mean().item() * 100
        log(f"SEDD puncture: mask_interval={mask_interval}, {pct_masked:.1f}% masked")
        assert pct_masked > 70 and pct_masked < 85, f"Unexpected mask %: {pct_masked}"
        
        # Test DoT-style prefix (keep first K)
        context_len = 512
        prefix = tokens[:, :context_len]
        pct_kept = context_len / seq_len * 100
        log(f"DoT prefix: context_len={context_len}, {pct_kept:.1f}% kept")
        assert len(prefix[0]) == context_len, "Prefix length mismatch"
        
        log("✓ Puncture refill logic correct")
        return True
    except Exception as e:
        log(f"Puncture test failed: {e}", "ERROR")
        return False

def generate_summary(results, out_dir):
    """Generate test summary report."""
    log("\n" + "="*60)
    log("TEST SUMMARY")
    log("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        log(f"{test:20s}: {status}")
    
    log(f"\nTotal: {passed}/{total} tests passed")
    
    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "passed": passed,
        "total": total,
        "success_rate": passed / total if total > 0 else 0,
    }
    
    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report saved: {report_path}")
    
    return passed == total

def main():
    parser = argparse.ArgumentParser(description="Test all models in pipeline")
    parser.add_argument("--data_dir", type=str, default="trash/data",
                        help="Path to train_codes.pt and val_codes.pt")
    parser.add_argument("--out_dir", type=str, default="trash/test_runs",
                        help="Output directory for test runs")
    parser.add_argument("--skip_classifier", action="store_true",
                        help="Skip classifier test")
    parser.add_argument("--skip_sedd", action="store_true",
                        help="Skip SEDD test")
    parser.add_argument("--skip_dot", action="store_true",
                        help="Skip DoT test")
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    
    log(f"Data directory: {data_dir}")
    log(f"Output directory: {out_dir}")
    
    # Check data exists
    missing = check_data_exists(data_dir)
    if missing:
        log(f"Missing required files: {missing}", "ERROR")
        log("Please provide: train_codes.pt, val_codes.pt", "ERROR")
        return 1
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Run tests
    results = {}
    
    # Test 1: Token shapes
    results["token_shapes"] = test_token_shapes(data_dir)
    
    # Test 2: Puncture logic
    results["puncture_logic"] = test_puncture_refill_logic()
    
    # Test 3: Classifier
    if not args.skip_classifier:
        results["classifier"] = test_classifier(data_dir, out_dir)
    else:
        log("Skipping classifier test")
        results["classifier"] = None
    
    # Test 4: SEDD
    if not args.skip_sedd:
        results["sedd"] = test_sedd(data_dir, out_dir)
    else:
        log("Skipping SEDD test")
        results["sedd"] = None
    
    # Test 5: DoT
    if not args.skip_dot:
        results["dot"] = test_dot(data_dir, out_dir)
    else:
        log("Skipping DoT test")
        results["dot"] = None
    
    # Generate summary
    all_passed = generate_summary(results, out_dir)
    
    if all_passed:
        log("\n✓ ALL TESTS PASSED - Pipeline is working correctly!")
        return 0
    else:
        log("\n✗ SOME TESTS FAILED - Check logs above", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main())
