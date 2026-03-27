#!/bin/bash
# run_verification.sh
# Complete pipeline to verify SEDD class conditioning by training a classifier on generated tokens
# Usage: bash run_verification.sh

set -e

export PYTHONUNBUFFERED=1

PYTHON=/data/joshi/MESHGPT/src/meshgpt_env/bin/python3
VERIFY=/data/joshi/tmp/MeshGeneration/conditional_verification
RUNS=/data/joshi/tmp/MeshGeneration/runs
OUT=$RUNS/classifier_eval

# SEDD checkpoints (update these paths after training completes)
CONDITIONAL_CKPT="$RUNS/sedd_conditional/checkpoints/last.ckpt"
UNCONDITIONAL_CKPT="$RUNS/sedd_unconditional/checkpoints/last.ckpt"

mkdir -p $OUT

echo "============================================================"
echo "  SEDD Conditional Verification Pipeline"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── Step 1: Generate tokens from CONDITIONAL SEDD ────────────────────────
echo "[1/6] Generating tokens from CONDITIONAL SEDD..."
echo "      Checkpoint: $CONDITIONAL_CKPT"

if [ ! -f "$CONDITIONAL_CKPT" ]; then
    echo "ERROR: Conditional SEDD checkpoint not found!"
    echo "       Expected: $CONDITIONAL_CKPT"
    echo "       Please ensure SEDD training completed successfully."
    exit 1
fi

$PYTHON -m torch.distributed.run --nproc_per_node=8 $VERIFY/generate_tokens_parallel.py \
    --mode conditional \
    --ckpt $CONDITIONAL_CKPT \
    --out_dir $OUT \
    --n_samples_per_class 100 \
    2>&1 | tee $OUT/generate_conditional.log

echo "[1/6] ✓ Conditional tokens saved"
echo ""

# ── Step 2: Generate tokens from UNCONDITIONAL SEDD ──────────────────────
echo "[2/6] Generating tokens from UNCONDITIONAL SEDD..."
echo "      Checkpoint: $UNCONDITIONAL_CKPT"

if [ ! -f "$UNCONDITIONAL_CKPT" ]; then
    echo "ERROR: Unconditional SEDD checkpoint not found!"
    echo "       Expected: $UNCONDITIONAL_CKPT"
    exit 1
fi

$PYTHON -m torch.distributed.run --nproc_per_node=8 $VERIFY/generate_tokens_parallel.py \
    --mode unconditional \
    --ckpt $UNCONDITIONAL_CKPT \
    --out_dir $OUT \
    --n_samples 4000 \
    2>&1 | tee $OUT/generate_unconditional.log

echo "[2/6] ✓ Unconditional tokens saved"
echo ""

# ── Step 3: Train classifier on CONDITIONAL tokens ───────────────────────
echo "[3/6] Training classifier on CONDITIONAL tokens (8 GPUs)..."

$PYTHON $VERIFY/train_classifier.py \
    --tokens_path $OUT/conditional_tokens.pt \
    --mode conditional \
    --out_dir $OUT/conditional_classifier \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --gpus 8 \
    2>&1 | tee $OUT/train_conditional.log

echo "[3/6] ✓ Conditional classifier trained"
echo ""

# ── Step 4: Train classifier on UNCONDITIONAL tokens ─────────────────────
echo "[4/6] Training classifier on UNCONDITIONAL tokens (8 GPUs)..."

$PYTHON $VERIFY/train_classifier.py \
    --tokens_path $OUT/unconditional_tokens.pt \
    --mode unconditional \
    --out_dir $OUT/unconditional_classifier \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --gpus 8 \
    2>&1 | tee $OUT/train_unconditional.log

echo "[4/6] ✓ Unconditional classifier trained"
echo ""

# ── Step 5: Evaluate CONDITIONAL classifier ──────────────────────────────
echo "[5/6] Evaluating CONDITIONAL classifier..."

COND_CKPT=$(ls -t $OUT/conditional_classifier/checkpoints/*.ckpt | head -1)
$PYTHON $VERIFY/evaluate.py \
    --ckpt $COND_CKPT \
    --tokens_path $OUT/conditional_tokens.pt \
    --mode conditional \
    --out_dir $OUT/conditional_classifier \
    2>&1 | tee $OUT/eval_conditional.log

echo "[5/6] ✓ Conditional evaluation complete"
echo ""

# ── Step 6: Evaluate UNCONDITIONAL classifier ────────────────────────────
echo "[6/6] Evaluating UNCONDITIONAL classifier..."

UNCOND_CKPT=$(ls -t $OUT/unconditional_classifier/checkpoints/*.ckpt | head -1)
$PYTHON $VERIFY/evaluate.py \
    --ckpt $UNCOND_CKPT \
    --tokens_path $OUT/unconditional_tokens.pt \
    --mode unconditional \
    --out_dir $OUT/unconditional_classifier \
    2>&1 | tee $OUT/eval_unconditional.log

echo "[6/6] ✓ Unconditional evaluation complete"
echo ""

# ── Generate final comparison report ─────────────────────────────────────
echo "============================================================"
echo "  FINAL RESULTS"
echo "============================================================"
echo ""

COND_ACC=$(grep -oP '"overall_accuracy": \K[0-9.]+' $OUT/conditional_classifier/eval_results.json | head -1)
UNCOND_ACC=$(grep -oP '"overall_accuracy": \K[0-9.]+' $OUT/unconditional_classifier/eval_results.json | head -1)

echo "Conditional Classifier Accuracy:   ${COND_ACC} ($(python3 -c "print(f'{float('$COND_ACC')*100:.2f}%')"))"
echo "Unconditional Classifier Accuracy: ${UNCOND_ACC} ($(python3 -c "print(f'{float('$UNCOND_ACC')*100:.2f}%')"))"
echo "Chance Level (40 classes):         0.025 (2.5%)"
echo ""

if (( $(echo "$COND_ACC > 0.7" | bc -l) )); then
    echo "✓ CONCLUSION: Class conditioning WORKS! Conditional tokens encode class information."
else
    echo "✗ CONCLUSION: Class conditioning INEFFECTIVE. Conditional tokens lack class structure."
fi

echo ""
echo "All results saved to: $OUT/"
echo "  - Tokens:     conditional_tokens.pt, unconditional_tokens.pt"
echo "  - Classifiers: conditional_classifier/, unconditional_classifier/"
echo "  - Plots:      */plots/ (confusion matrix, t-SNE, per-class accuracy)"
echo ""
echo "Finished: $(date)"
echo "============================================================"
