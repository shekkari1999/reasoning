#!/usr/bin/env bash
# Memory analysis on 2× GPU before / alongside training.
# Produces JSON reports in results/ for README memory table.
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
echo "=== 1/3 Eval memory (1 GPU, inference only) ==="
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset gsm8k \
  --max_samples 20 \
  --stage base \
  --batch_size 16
echo "  -> results/base_eval_summary.json includes memory.peak_gb_max"

echo ""
echo "=== 2/3 SFT memory sweep (2 GPU, 10 steps) ==="
torchrun --nproc_per_node=2 src/sweep.py --mode quick --num_steps 10
echo "  -> results/sweep_quick_results.json"

echo ""
echo "=== 3/3 GRPO memory sweep (2 GPU, policy + ref model) ==="
torchrun --nproc_per_node=2 src/sweep.py --mode grpo
echo "  -> results/sweep_grpo_results.json"

echo ""
echo "=== Memory reports ==="
ls -la results/memory_*.json results/sweep_*_results.json 2>/dev/null || true
echo ""
echo "After real training, also check:"
echo "  results/memory_sft.json"
echo "  results/memory_dr_grpo.json / memory_grpo.json / memory_dapo.json"
echo "  checkpoints/*/sft_metrics.json (peak_mem_gb over steps)"
