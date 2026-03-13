#!/usr/bin/env python3
"""
verify_files.py - Verify all necessary files are copied for GitHub push
"""

import os
import sys

def check_file(path, description):
    """Check if file exists and print status."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_dir(path, description):
    """Check if directory exists and count files."""
    exists = os.path.exists(path)
    if exists and os.path.isdir(path):
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        print(f"✓ {description}: {path} ({count} files)")
        return True
    else:
        print(f"✗ {description}: {path}")
        return False

def main():
    base = "/data/joshi/MESHGPT/push_github_src"
    
    print("=" * 70)
    print("VERIFYING MESHGPT GITHUB REPOSITORY STRUCTURE")
    print("=" * 70)
    
    all_ok = True
    
    # Main README
    print("\n[Main Repository]")
    all_ok &= check_file(f"{base}/README.md", "Main README")
    
    # MeshGPT VQ-VAE
    print("\n[MeshGPT VQ-VAE Module]")
    all_ok &= check_file(f"{base}/mesh_vqvae/README.md", "README")
    all_ok &= check_file(f"{base}/mesh_vqvae/requirements.txt", "Requirements")
    all_ok &= check_file(f"{base}/mesh_vqvae/extract_codes.py", "Extract codes script")
    all_ok &= check_dir(f"{base}/mesh_vqvae/src", "Source directory")
    all_ok &= check_dir(f"{base}/mesh_vqvae/training_scripts", "Training scripts")
    
    print("\n[MeshGPT Core Source Files]")
    src_files = [
        "config.py", "model.py", "encoder.py", "decoder.py", "quantizer.py",
        "masker.py", "demasker.py", "heads.py", "dataset.py", "preprocessing.py",
        "train.py", "evaluate.py", "stage3_decode.py"
    ]
    for f in src_files:
        all_ok &= check_file(f"{base}/mesh_vqvae/src/{f}", f)
    
    print("\n[MeshGPT Training Scripts]")
    all_ok &= check_file(f"{base}/mesh_vqvae/training_scripts/train_with_classifier.py", 
                         "Train with classifier")
    all_ok &= check_file(f"{base}/mesh_vqvae/training_scripts/train_without_classifier.py", 
                         "Train without classifier")
    
    # SEDD Diffusion
    print("\n[SEDD Diffusion Module]")
    all_ok &= check_file(f"{base}/diffusion_model/README.md", "README")
    all_ok &= check_file(f"{base}/diffusion_model/requirements.txt", "Requirements")
    all_ok &= check_file(f"{base}/diffusion_model/SEDD.py", "SEDD implementation")
    all_ok &= check_file(f"{base}/diffusion_model/train_sedd.py", "SEDD training script")
    all_ok &= check_file(f"{base}/diffusion_model/preprocessing.py", "Preprocessing utilities")
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ ALL FILES VERIFIED - Ready for GitHub push!")
    else:
        print("✗ SOME FILES MISSING - Please check above")
        sys.exit(1)
    print("=" * 70)
    
    # File counts
    print("\n[Statistics]")
    mesh_py = len([f for f in os.listdir(f"{base}/mesh_vqvae/src") if f.endswith('.py')])
    train_py = len([f for f in os.listdir(f"{base}/mesh_vqvae/training_scripts") if f.endswith('.py')])
    sedd_py = len([f for f in os.listdir(f"{base}/diffusion_model") if f.endswith('.py')])
    
    print(f"  MeshGPT source files: {mesh_py}")
    print(f"  MeshGPT training scripts: {train_py}")
    print(f"  SEDD files: {sedd_py}")
    print(f"  Total Python files: {mesh_py + train_py + sedd_py + 1}")  # +1 for extract_codes.py
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
