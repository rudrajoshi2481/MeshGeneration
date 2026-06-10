#!/usr/bin/env python3
"""
test_plots.py
-------------
Test script to verify that all models generate plots correctly.
Runs small-scale training on all three models and checks for plot generation.

Usage:
    python test_plots.py

Outputs:
    trash/test_plots/<model>/plots/ - generated plots
    trash/test_plots/summary.json - verification results
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

# Base paths
BASE_DIR = "/media/rudhra/ChenLabData1/Middleware/Nvidia_project/sementic_channel_project/MeshGeneration"
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "trash", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "trash", "test_plots")

def ensure_data_exists():
    """Check if data files exist."""
    train_path = os.path.join(DATA_DIR, "train_codes.pt")
    val_path = os.path.join(DATA_DIR, "val_codes.pt")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"[ERROR] Data files not found at {DATA_DIR}")
        print(f"        Expected: train_codes.pt, val_codes.pt")
        return False
    
    print(f"[OK] Data files found: {DATA_DIR}")
    return True

def run_classifier_test():
    """Test classifier with enhanced plotting."""
    print("\n" + "="*60)
    print("  Testing Classifier with Enhanced Plotting")
    print("="*60)
    
    out_dir = os.path.join(OUTPUT_DIR, "classifier")
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "models", "classifier", "train_classifier.py"),
        "--tokens_path", os.path.join(DATA_DIR, "train_codes.pt"),
        "--mode", "conditional",
        "--out_dir", out_dir,
        "--epochs", "3",
        "--batch_size", "16",
        "--gpus", "1",
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("[STDERR]", result.stderr)
        
        # Check for plots
        plot_dir = os.path.join(out_dir, "plots")
        expected_plots = [
            "training_curves.png",
            "confusion_matrix.png",
            "per_class_accuracy.png"
        ]
        
        results = {"plot_dir": plot_dir, "plots_found": [], "plots_missing": []}
        
        for plot in expected_plots:
            plot_path = os.path.join(plot_dir, plot)
            if os.path.exists(plot_path):
                results["plots_found"].append(plot)
                print(f"  [OK] {plot}")
            else:
                results["plots_missing"].append(plot)
                print(f"  [MISSING] {plot}")
        
        # Check for classification report
        report_path = os.path.join(out_dir, "classification_report.txt")
        if os.path.exists(report_path):
            results["plots_found"].append("classification_report.txt")
            print(f"  [OK] classification_report.txt")
        else:
            results["plots_missing"].append("classification_report.txt")
            print(f"  [MISSING] classification_report.txt")
        
        results["success"] = len(results["plots_missing"]) == 0
        return results
        
    except subprocess.TimeoutExpired:
        print("[ERROR] Classifier test timed out")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"[ERROR] Classifier test failed: {e}")
        return {"success": False, "error": str(e)}

def run_sedd_test():
    """Test SEDD with plotting."""
    print("\n" + "="*60)
    print("  Testing SEDD with Plotting")
    print("="*60)
    
    out_dir = os.path.join(OUTPUT_DIR, "sedd")
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "models", "diffusion", "train_sedd.py"),
        "--mode", "small",
        "--n_train", "100",
        "--epochs", "5",
        "--batch_size", "8",
        "--gpus", "1",
    ]
    
    # Need to pass run_dir differently since train_sedd.py constructs it internally
    # We'll just check the most recent run in sedd_runs
    
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=BASE_DIR)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)  # Last 2000 chars
        if result.stderr:
            print("[STDERR]", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        
        # Find the most recent sedd run
        sedd_runs = os.path.join(os.path.dirname(BASE_DIR), "trash", "sedd_runs")
        if os.path.exists(sedd_runs):
            runs = sorted([d for d in os.listdir(sedd_runs) if d.startswith("sedd_small_")])
            if runs:
                latest_run = os.path.join(sedd_runs, runs[-1])
                plot_dir = os.path.join(latest_run, "plots")
                
                expected_plots = [
                    "curves_ep0000.png",
                    "gen_hist_ep0000.png",
                    "code_dist_ep0000.png"
                ]
                
                results = {"plot_dir": plot_dir, "plots_found": [], "plots_missing": []}
                
                for plot in expected_plots:
                    plot_path = os.path.join(plot_dir, plot)
                    if os.path.exists(plot_path):
                        results["plots_found"].append(plot)
                        print(f"  [OK] {plot}")
                    else:
                        results["plots_missing"].append(plot)
                        print(f"  [MISSING] {plot}")
                
                results["success"] = len(results["plots_found"]) > 0  # At least some plots
                return results
        
        return {"success": False, "error": "No sedd run found"}
        
    except subprocess.TimeoutExpired:
        print("[ERROR] SEDD test timed out")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"[ERROR] SEDD test failed: {e}")
        return {"success": False, "error": str(e)}

def run_dot_test():
    """Test DoT with plotting."""
    print("\n" + "="*60)
    print("  Testing DoT with Plotting")
    print("="*60)
    
    out_dir = os.path.join(OUTPUT_DIR, "dot")
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "models", "autoregressive", "train_dot_mesh.py"),
        "--mode", "small",
        "--data_dir", DATA_DIR,
        "--out_base", out_dir,
        "--epochs", "5",
        "--batch_size", "4",
        "--gpus", "1",
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print("[STDERR]", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        
        # Find the most recent dot run
        if os.path.exists(out_dir):
            runs = sorted([d for d in os.listdir(out_dir) if d.startswith("dot_run_")])
            if runs:
                latest_run = os.path.join(out_dir, runs[-1])
                plot_dir = os.path.join(latest_run, "plots")
                
                expected_plots = [
                    "curves_ep0000.png",
                    "gen_hist_ep0000.png",
                    "code_dist_ep0000.png"
                ]
                
                results = {"plot_dir": plot_dir, "plots_found": [], "plots_missing": []}
                
                for plot in expected_plots:
                    plot_path = os.path.join(plot_dir, plot)
                    if os.path.exists(plot_path):
                        results["plots_found"].append(plot)
                        print(f"  [OK] {plot}")
                    else:
                        results["plots_missing"].append(plot)
                        print(f"  [MISSING] {plot}")
                
                results["success"] = len(results["plots_found"]) > 0
                return results
        
        return {"success": False, "error": "No dot run found"}
        
    except subprocess.TimeoutExpired:
        print("[ERROR] DoT test timed out")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"[ERROR] DoT test failed: {e}")
        return {"success": False, "error": str(e)}

def main():
    print("\n" + "="*70)
    print("  PLOT GENERATION VERIFICATION TEST")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check data
    if not ensure_data_exists():
        print("\n[ABORT] Data files missing, cannot run tests")
        sys.exit(1)
    
    # Run tests
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": OUTPUT_DIR,
        "tests": {}
    }
    
    # Test Classifier
    results["tests"]["classifier"] = run_classifier_test()
    
    # Test SEDD
    results["tests"]["sedd"] = run_sedd_test()
    
    # Test DoT
    results["tests"]["dot"] = run_dot_test()
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    all_passed = True
    for model, test_results in results["tests"].items():
        status = "PASS" if test_results.get("success") else "FAIL"
        all_passed = all_passed and test_results.get("success", False)
        print(f"  {model.upper():12s}: {status}")
        if "plots_found" in test_results:
            print(f"              Plots found: {len(test_results['plots_found'])}")
        if "plots_missing" in test_results and test_results["plots_missing"]:
            print(f"              Missing: {', '.join(test_results['plots_missing'])}")
        if "error" in test_results:
            print(f"              Error: {test_results['error']}")
    
    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Summary saved to: {summary_path}")
    print(f"[OK] All plots in: {OUTPUT_DIR}")
    
    if all_passed:
        print("\n[PASS] All plot generation tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Check summary.json for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
