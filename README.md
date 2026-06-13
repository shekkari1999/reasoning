# reasoning

Post-training pipeline for math reasoning on Qwen2.5-3B. Reimplements the DeepSeek-R1 workflow in raw PyTorch: SFT on chain-of-thought traces, then Dr.GRPO with verifiable rewards.

## What it does

- **SFT** on a 10k subset of OpenR1-Math CoT traces
- **Dr.GRPO** with group-relative advantages, clipped policy loss, and exact-match rewards (no critic model)
- **Eval** on GSM8K and MATH500 with a fixed CoT prompt harness

Training uses FSDP on 2× 96 GB Blackwell GPUs. Batch size and sequence length were swept before RL to reach 86% GPU utilization.

## Results

| Stage | GSM8K | MATH500 |
| --- | ---: | ---: |
| Base | 63.2% | 26.3% |
| SFT | 71.2% | 33.1% |
| Dr.GRPO | 78.2% | 45.6% |

## Stack

Python, PyTorch, FSDP, Hugging Face Transformers
