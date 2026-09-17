#!/usr/bin/env bash
# Memory analysis on two GPUs before or alongside training.
# Produces JSON reports under results/.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "=== GPU info ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo ""
echo "=== 1/2 Eval memory probe (20 samples, does NOT touch base results) ==="
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset gsm8k \
  --max_samples 20 \
  --stage memory_probe \
  --batch_size 16
echo "  -> results/memory_probe_eval_summary.json (memory.peak_gb_max)"

echo ""
echo "=== 2/2 SFT memory sweep (2 GPU, 10 steps) ==="
torchrun --nproc_per_node=2 src/sweep.py --mode quick --num_steps 10
echo "  -> results/sweep_quick_results.json"

echo ""
echo "=== Memory reports ==="
ls -la results/memory_*.json results/sweep_*_results.json 2>/dev/null || true
echo ""
echo "After real training, also check:"
echo "  results/memory_sft.json"
echo "  results/memory_dr_grpo.json"
echo "  checkpoints/sft/sft_metrics.json"
echo "  checkpoints/dr_grpo/dr_grpo_metrics.json"
