#!/bin/bash
# scripts/profile_grpo.sh — Profile GRPO baseline vs overlapped
set -e

mkdir -p profiles

echo "=== Profiling GRPO Baseline (no overlap) ==="
nsys profile \
    --trace=cuda,nvtx,nccl \
    --capture-range=cudaProfilerApi \
    --output=profiles/grpo_no_overlap \
    --force-overwrite=true \
    torchrun --nproc_per_node=2 src/rl_train.py \
        --algo grpo \
        --config configs/grpo_config.yaml \
        --profile \
        --num_steps 25

echo ""
echo "=== Profiling GRPO Overlapped ==="
nsys profile \
    --trace=cuda,nvtx,nccl \
    --capture-range=cudaProfilerApi \
    --output=profiles/grpo_overlapped \
    --force-overwrite=true \
    torchrun --nproc_per_node=2 src/rl_train_overlapped.py \
        --config configs/grpo_config.yaml \
        --profile \
        --num_steps 25

echo ""
echo "=== Done. Compare these side by side in Nsight Systems GUI ==="
echo "  profiles/grpo_no_overlap.nsys-rep"
echo "  profiles/grpo_overlapped.nsys-rep"
echo ""
echo "  Look for: async_next_rollout running concurrently with policy_update"
