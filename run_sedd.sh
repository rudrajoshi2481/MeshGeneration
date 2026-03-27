#!/bin/bash
# run_sedd.sh — Extract fresh codes, then train unconditional + conditional SEDD sequentially
# Usage: bash run_sedd.sh

set -e

export PYTHONUNBUFFERED=1

PYTHON=/data/joshi/MESHGPT/src/meshgpt_env/bin/python3
DIFFUSION=/data/joshi/tmp/MeshGeneration/diffusion_model
RUNS=/data/joshi/tmp/MeshGeneration/runs
SEDD_DATA=$RUNS/sedd_data
LOG=$RUNS/sedd_launch.log

mkdir -p $SEDD_DATA

echo "=============================================="
echo "  SEDD Pipeline"
echo "  Started: $(date)"
echo "=============================================="

# ── Step 1: Extract fresh codes ──────────────────────────────────────────────
echo ""
echo "[1/3] Extracting fresh codes from latest mesh_vqvae checkpoint..."
$PYTHON $DIFFUSION/extract_fresh_codes.py \
    --out_dir $SEDD_DATA \
    --overwrite \
    2>&1 | tee $RUNS/extract_codes.log
echo "[1/3] Done: $SEDD_DATA"

# ── Step 2: Train UNCONDITIONAL SEDD ─────────────────────────────────────────
echo ""
echo "[2/3] Training UNCONDITIONAL SEDD (no class conditioning)..."
$PYTHON $DIFFUSION/train_sedd_enhanced.py \
    --mode unconditional \
    --gpus 8 \
    2>&1 | tee $RUNS/sedd_unconditional.log
echo "[2/3] Unconditional training complete."

# ── Step 3: Train CONDITIONAL SEDD ───────────────────────────────────────────
echo ""
echo "[3/3] Training CONDITIONAL SEDD (with class conditioning)..."
$PYTHON $DIFFUSION/train_sedd_enhanced.py \
    --mode conditional \
    --gpus 8 \
    2>&1 | tee $RUNS/sedd_conditional.log
echo "[3/3] Conditional training complete."

echo ""
echo "=============================================="
echo "  All SEDD runs complete!"
echo "  Results: $RUNS/sedd_unconditional_*"
echo "           $RUNS/sedd_conditional_*"
echo "  Finished: $(date)"
echo "=============================================="
