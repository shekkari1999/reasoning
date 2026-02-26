#!/bin/bash
# scripts/run_dr_grpo.sh
set -e

torchrun --nproc_per_node=2 src/rl_train.py \
    --algo dr_grpo --config configs/dr_grpo_config.yaml
