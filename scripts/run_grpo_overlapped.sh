#!/bin/bash
# scripts/run_grpo_overlapped.sh
set -e

torchrun --nproc_per_node=2 src/rl_train_overlapped.py \
    --config configs/grpo_config.yaml
