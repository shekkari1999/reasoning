# reasoning

Small-scale math post-training pipeline for Qwen2.5-3B, implemented in raw
PyTorch. It runs completion-only SFT followed by Dr.GRPO with verifiable rewards.

## What it does

- **SFT** on a 10k subset of OpenR1-Math CoT traces
- **Dr.GRPO** with group-relative advantages, token-level clipped ratios, and a fixed generation-length denominator
- **Eval** on GSM8K and MATH500 with one frozen prompt and decoding harness
- **Provenance** through seeded dataset manifests, per-example evaluator outputs, and an evaluator-built summary
- **Reproducibility** through pinned Hugging Face model and dataset revisions

Training uses FSDP on two GPUs with bf16 mixed precision and activation
checkpointing.

## Results

| Stage | GSM8K | MATH500 |
| --- | ---: | ---: |
| Base | 63.2% | 26.3% |
| SFT | 71.2% | 33.1% |
| Dr.GRPO | 78.2% | 45.6% |


## Stack

Python, PyTorch, FSDP, Hugging Face Transformers
