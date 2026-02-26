#!/bin/bash
# scripts/run_dapo.sh
set -e

torchrun --nproc_per_node=2 src/rl_train.py \
    --algo dapo --config configs/dapo_config.yaml
