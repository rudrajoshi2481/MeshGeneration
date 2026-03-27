#!/bin/bash
# retest_no_class_token.sh
# Proper test: Remove class tokens from generated sequences to verify if SEDD
# truly encodes class information in the token patterns themselves
# Usage: bash retest_no_class_token.sh

set -e

export PYTHONUNBUFFERED=1

PYTHON=/data/joshi/MESHGPT/src/meshgpt_env/bin/python3
VERIFY=/data/joshi/tmp/MeshGeneration/conditional_verification
RUNS=/data/joshi/tmp/MeshGeneration/runs
OUT=$RUNS/classifier_eval_no_class_token

# Use existing tokens from previous run
CONDITIONAL_TOKENS="$RUNS/classifier_eval/conditional_tokens.pt"
UNCONDITIONAL_TOKENS="$RUNS/classifier_eval/unconditional_tokens.pt"

mkdir -p $OUT

echo "============================================================"
echo "  SEDD Verification (No Class Token Test)"
echo "  Testing if class info is in token patterns, not just class token"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── Step 1: Strip class tokens from sequences ────────────────────────────
echo "[1/5] Stripping class tokens from conditional sequences..."

$PYTHON $VERIFY/strip_class_tokens.py \
    --tokens_path $CONDITIONAL_TOKENS \
    --out_path $OUT/conditional_tokens_no_class.pt \
    2>&1 | tee $OUT/strip_conditional.log

echo "[1/5] ✓ Conditional tokens stripped"
echo ""

echo "[2/5] Copying unconditional tokens (no class token to strip)..."
cp $UNCONDITIONAL_TOKENS $OUT/unconditional_tokens_no_class.pt
echo "[2/5] ✓ Unconditional tokens copied"
echo ""

# ── Step 2: Train classifier on CONDITIONAL tokens (no class) ────────────
echo "[3/5] Training classifier on CONDITIONAL tokens WITHOUT class token (8 GPUs)..."

$PYTHON $VERIFY/train_classifier.py \
    --tokens_path $OUT/conditional_tokens_no_class.pt \
    --mode conditional_no_class \
    --out_dir $OUT/conditional_classifier \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --gpus 8 \
    2>&1 | tee $OUT/train_conditional.log

echo "[3/5] ✓ Conditional classifier trained"
echo ""

# ── Step 3: Train classifier on UNCONDITIONAL tokens ─────────────────────
echo "[4/5] Training classifier on UNCONDITIONAL tokens (8 GPUs)..."

$PYTHON $VERIFY/train_classifier.py \
    --tokens_path $OUT/unconditional_tokens_no_class.pt \
    --mode unconditional_no_class \
    --out_dir $OUT/unconditional_classifier \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --gpus 8 \
    2>&1 | tee $OUT/train_unconditional.log

echo "[4/5] ✓ Unconditional classifier trained"
echo ""

# ── Step 4: Evaluate both classifiers ────────────────────────────────────
echo "[5/5] Evaluating classifiers..."

COND_CKPT=$(ls -t $OUT/conditional_classifier/checkpoints/*.ckpt | head -1)
$PYTHON $VERIFY/evaluate.py \
    --ckpt $COND_CKPT \
    --tokens_path $OUT/conditional_tokens_no_class.pt \
    --mode conditional_no_class \
    --out_dir $OUT/conditional_classifier \
    2>&1 | tee $OUT/eval_conditional.log

UNCOND_CKPT=$(ls -t $OUT/unconditional_classifier/checkpoints/*.ckpt | head -1)
$PYTHON $VERIFY/evaluate.py \
    --ckpt $UNCOND_CKPT \
    --tokens_path $OUT/unconditional_tokens_no_class.pt \
    --mode unconditional_no_class \
    --out_dir $OUT/unconditional_classifier \
    2>&1 | tee $OUT/eval_unconditional.log

echo "[5/5] ✓ Evaluation complete"
echo ""

# ── Generate comparison report ───────────────────────────────────────────
echo "============================================================"
echo "  RESULTS (No Class Token Test)"
echo "============================================================"
echo ""

COND_ACC=$(grep -oP '"overall_accuracy": \K[0-9.]+' $OUT/conditional_classifier/eval_results.json | head -1)
UNCOND_ACC=$(grep -oP '"overall_accuracy": \K[0-9.]+' $OUT/unconditional_classifier/eval_results.json | head -1)

echo "Conditional Classifier Accuracy (no class token):   ${COND_ACC} ($(awk "BEGIN {printf \"%.2f\", $COND_ACC*100}")%)"
echo "Unconditional Classifier Accuracy:                  ${UNCOND_ACC} ($(awk "BEGIN {printf \"%.2f\", $UNCOND_ACC*100}")%)"
echo "Chance Level (40 classes):                          0.025 (2.5%)"
echo ""

# Compare with original results
ORIG_COND_ACC=$(grep -oP '"overall_accuracy": \K[0-9.]+' $RUNS/classifier_eval/conditional_classifier/eval_results.json | head -1)
echo "============================================================"
echo "  COMPARISON"
echo "============================================================"
echo ""
echo "Original Test (WITH class token):    ${ORIG_COND_ACC} ($(awk "BEGIN {printf \"%.2f\", $ORIG_COND_ACC*100}")%)"
echo "New Test (WITHOUT class token):      ${COND_ACC} ($(awk "BEGIN {printf \"%.2f\", $COND_ACC*100}")%)"
echo ""

if (( $(echo "$COND_ACC > 0.5" | bc -l) )); then
    echo "✓ CONCLUSION: Class conditioning WORKS! Tokens encode class information."
    echo "  Even without the class token, classifier achieves >50% accuracy."
else
    echo "✗ CONCLUSION: Class conditioning was CHEATING via class token."
    echo "  Without class token, accuracy drops to near-chance level."
fi

echo ""
echo "All results saved to: $OUT/"
echo "Finished: $(date)"
echo "============================================================"
