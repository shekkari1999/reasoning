#!/bin/bash
# scripts/run_eval.sh — Evaluate all checkpoints on GSM8K and MATH500
set -e

echo "=== Evaluating all checkpoints ==="

# Base model
echo "--- Base Qwen-2.5-3B ---"
python src/baseline_eval.py \
    --model Qwen/Qwen2.5-3B \
    --dataset both --stage base --batch_size 16

# SFT
if [ -d "checkpoints/sft/final" ]; then
    echo "--- SFT ---"
    python src/baseline_eval.py \
        --model checkpoints/sft/final \
        --dataset both --stage sft --prompt_mode sft --batch_size 16
fi

# GRPO
if [ -d "checkpoints/grpo/final" ]; then
    echo "--- GRPO ---"
    python src/baseline_eval.py \
        --model checkpoints/grpo/final \
        --dataset both --stage grpo --prompt_mode sft --batch_size 16
fi

# Dr. GRPO
if [ -d "checkpoints/dr_grpo/final" ]; then
    echo "--- Dr. GRPO ---"
    python src/baseline_eval.py \
        --model checkpoints/dr_grpo/final \
        --dataset both --stage dr_grpo --prompt_mode sft --batch_size 16
fi

# DAPO
if [ -d "checkpoints/dapo/final" ]; then
    echo "--- DAPO ---"
    python src/baseline_eval.py \
        --model checkpoints/dapo/final \
        --dataset both --stage dapo --prompt_mode sft --batch_size 16
fi

echo ""
echo "=== All evaluations complete. Results in results/ ==="
echo "  Compare: results/*_eval_summary.json"
