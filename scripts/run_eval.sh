#!/usr/bin/env bash
# scripts/run_eval.sh — Evaluate all checkpoints on GSM8K and MATH500
set -euo pipefail

cd "$(dirname "$0")/.."

SFT_CHECKPOINT="checkpoints/sft/final/step_625"
DR_GRPO_CHECKPOINT="checkpoints/dr_grpo/final/step_200"
MODEL_REVISION="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

echo "=== Evaluating all checkpoints ==="

# Base model
echo "--- Base Qwen-2.5-3B ---"
python src/baseline_eval.py \
    --model Qwen/Qwen2.5-3B \
    --model_revision "$MODEL_REVISION" \
    --dataset both --stage base --prompt_mode sft --batch_size 16

# SFT
if [ -d "$SFT_CHECKPOINT" ]; then
    echo "--- SFT ---"
    python src/baseline_eval.py \
        --model "$SFT_CHECKPOINT" \
        --dataset both --stage sft --prompt_mode sft --batch_size 16
fi

# Dr. GRPO
if [ -d "$DR_GRPO_CHECKPOINT" ]; then
    echo "--- Dr. GRPO ---"
    python src/baseline_eval.py \
        --model "$DR_GRPO_CHECKPOINT" \
        --dataset both --stage dr_grpo --prompt_mode sft --batch_size 16
fi

if [ -f results/base_eval_summary.json ] && \
   [ -f results/sft_eval_summary.json ] && \
   [ -f results/dr_grpo_eval_summary.json ]; then
    python src/aggregate_results.py
fi

echo ""
echo "=== All evaluations complete. Results in results/ ==="
echo "  Canonical summary: results/latest_run_summary.json"
