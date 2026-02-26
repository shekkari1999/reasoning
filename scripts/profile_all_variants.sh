#!/bin/bash
# scripts/profile_all_variants.sh — Profile all 3 RL variants for comparison
set -e

mkdir -p profiles
STEPS=20

echo "=== Profiling all RL variants ($STEPS steps each) ==="

for ALGO in grpo dr_grpo dapo; do
    echo ""
    echo "--- Profiling $ALGO ---"
    nsys profile \
        --trace=cuda,nvtx,nccl \
        --capture-range=cudaProfilerApi \
        --output=profiles/${ALGO}_profile \
        --force-overwrite=true \
        torchrun --nproc_per_node=2 src/rl_train.py \
            --algo $ALGO \
            --config configs/${ALGO}_config.yaml \
            --profile \
            --num_steps $STEPS
done

echo ""
echo "=== All profiles generated ==="
echo "  profiles/grpo_profile.nsys-rep"
echo "  profiles/dr_grpo_profile.nsys-rep"
echo "  profiles/dapo_profile.nsys-rep"
echo ""
echo "  Compare step times, NCCL overhead, memory patterns across variants."
echo "  Dr.GRPO and DAPO should show lower memory (no ref model)."
