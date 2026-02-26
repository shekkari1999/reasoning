#!/bin/bash
# scripts/profile_ncu.sh — Nsight Compute kernel analysis (single GPU only)
set -e

mkdir -p profiles

echo "=== Nsight Compute — Single GPU kernel profiling ==="
echo "  NOTE: ncu profiles single-GPU only. Do NOT use with torchrun."
echo ""

# Profile dominant GEMMs in a single forward+backward step
echo "--- Profiling SFT GEMMs ---"
ncu --set full \
    --kernel-name regex:"gemm|xmma|cutlass" \
    --launch-count 10 \
    --output profiles/sft_kernel_analysis \
    python -c "
import torch
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', torch_dtype=torch.bfloat16, device_map='cuda')
x = torch.randint(0, 151936, (2, 512), device='cuda')
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = model(input_ids=x)
    loss = F.cross_entropy(out.logits[:, :-1, :].reshape(-1, 151936), x[:, 1:].reshape(-1))
loss.backward()
torch.cuda.synchronize()
print('Done.')
"

echo ""
echo "=== Done. Open profiles/sft_kernel_analysis.ncu-rep in Nsight Compute GUI ==="
echo "  Check: occupancy, memory throughput, compute throughput, tensor core utilization"
