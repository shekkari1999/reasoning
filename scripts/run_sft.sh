#!/usr/bin/env bash
# scripts/run_sft.sh
set -euo pipefail

cd "$(dirname "$0")/.."

torchrun --nproc_per_node=2 src/sft_train.py \
    --config configs/sft_config.yaml
