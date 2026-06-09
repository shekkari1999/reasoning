#!/usr/bin/env bash
# Run once on a fresh GPU instance after smoke_test.py passes.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== smoke test ==="
python tests/smoke_test.py

echo ""
echo "=== quick eval (20 GSM8K) ==="
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset gsm8k \
  --max_samples 20 \
  --stage base \
  --batch_size 8

echo ""
echo "=== full baseline eval ==="
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset both \
  --stage base \
  --batch_size 16

echo ""
echo "=== pass@k analysis ==="
python src/baseline_analysis.py \
  --model Qwen/Qwen2.5-3B \
  --pass_k_samples 200

echo ""
echo "Done. Check results/"
