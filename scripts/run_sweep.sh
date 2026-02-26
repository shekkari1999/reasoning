#!/bin/bash
# scripts/run_sweep.sh — Parameter sweep to find best config
set -e

echo "=== SFT Parameter Sweep ==="
torchrun --nproc_per_node=2 src/sweep.py --mode quick --num_steps 10

echo ""
echo "=== GRPO Memory Sweep (2 models) ==="
torchrun --nproc_per_node=2 src/sweep.py --mode grpo
