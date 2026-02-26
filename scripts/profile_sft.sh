#!/bin/bash
# scripts/profile_sft.sh — Profile SFT training with Nsight Systems
set -e

mkdir -p profiles

echo "=== Profiling SFT (NVTX + NCCL traces) ==="
echo "  Warmup: 10 steps, Capture: 20 steps"

nsys profile \
    --trace=cuda,nvtx,nccl \
    --capture-range=cudaProfilerApi \
    --output=profiles/sft_profile \
    --force-overwrite=true \
    torchrun --nproc_per_node=2 src/sft_train.py \
        --config configs/sft_config.yaml \
        --profile \
        --num_steps 35

echo "=== Done. Open profiles/sft_profile.nsys-rep in Nsight Systems GUI ==="
