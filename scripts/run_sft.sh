#!/bin/bash
# scripts/run_sft.sh
set -e

torchrun --nproc_per_node=2 src/sft_train.py \
    --config configs/sft_config.yaml
