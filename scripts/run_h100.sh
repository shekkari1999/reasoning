#!/usr/bin/env bash
# One-shot setup + baseline eval on a fresh 1x H100 instance.
# Usage:  export HF_TOKEN=hf_...   (optional if already logged in)
#         bash scripts/run_h100.sh
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

echo "=== Reasoning_model — H100 baseline ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found"
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ -n "${HF_TOKEN:-}" ]; then
  if command -v hf &>/dev/null; then
    hf auth login --token "$HF_TOKEN"
  else
    echo "WARN: hf CLI not found; skipping HF login (model may already be cached)"
  fi
fi

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo ""
echo "=== 1/4 smoke test ==="
python tests/smoke_test.py

echo ""
echo "=== 2/4 quick eval (20 GSM8K, ~2 min) ==="
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset gsm8k \
  --max_samples 20 \
  --stage base \
  --batch_size 8

echo ""
echo "=== 3/4 full baseline (GSM8K + MATH500 test, ~45-90 min) ==="
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset both \
  --stage base \
  --batch_size 16

echo ""
echo "=== 4/4 pass@k (200 GSM8K samples, ~30-60 min) ==="
python src/baseline_analysis.py \
  --model Qwen/Qwen2.5-3B \
  --pass_k_samples 200

echo ""
echo "=== Done ==="
echo "Results:"
ls -la results/
cat results/base_eval_summary.json
