#!/bin/bash
# scripts/run_grpo.sh
set -e

torchrun --nproc_per_node=2 src/rl_train.py \
    --algo grpo --config configs/grpo_config.yaml
