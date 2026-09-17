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

These are the project owner's recorded metrics from the completed historical
run. Its raw evaluator outputs and checkpoints were not preserved in this
checkout, so the record is kept separately in
[`results/reported_run_summary.json`](results/reported_run_summary.json).
That record also preserves the reported 86% peak GPU-utilization figure while
making clear that its profiler artifact and exact metric definition are absent.

New runs produce `base_eval_summary.json`, `sft_eval_summary.json`, and
`dr_grpo_eval_summary.json`. `src/aggregate_results.py` validates those files
and generates `results/latest_run_summary.json`; that generated file is the
canonical artifact for future result claims.

## Stack

Python, PyTorch, FSDP, Hugging Face Transformers
