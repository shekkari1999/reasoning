# CLAUDE.md — Project Context for Claude Code

## What This Project Is

Training Qwen-2.5-3B into a reasoning model using raw PyTorch (no TRL, no HF Trainer). Two-stage pipeline: SFT on chain-of-thought traces, then RL (GRPO/Dr.GRPO/DAPO) to improve reasoning quality. Running on 2× RTX 4090 (24GB each, PCIe interconnect, no NVLink).

## Current Status

- **SFT: DONE** — 300 steps on OpenR1-Math-220k, checkpoint at `checkpoints/sft/final/step_300`. Loss converged to 2.2. Model outputs `<think>...</think><answer>...</answer>` format correctly. ~40% accuracy on GSM8K zero-shot (down from 70% few-shot baseline due to seq_len=512 truncation).
- **RL: BROKEN** — `rl_train.py` crashes silently when run with `torchrun --nproc_per_node=2`. Loads model, loads data, then dies at first training step with NCCL abort and no Python traceback. Error logs at `/workspace/dr_grpo_log.txt` and `/workspace/dr_grpo_error.txt`.
- **Baseline eval: DONE** — Base Qwen-2.5-3B scores 70% GSM8K, ~40% MATH500. Pass@8 = 95% (27.3% gap = RL opportunity).

## Architecture

```
configs/           — YAML configs for SFT, GRPO, Dr.GRPO, DAPO
src/
  model.py         — load_model, load_tokenizer, wrap_model_fsdp, load_reference_model, save_hf_checkpoint
  data.py          — SFTDataset, RLPromptDataset, create_sft_dataloader, create_rl_dataloader
  losses.py        — get_per_token_logprobs, sequence_logprobs, compute_advantages_*, grpo_loss, dr_grpo_loss, dapo_loss
  rewards.py       — extract_answer, compute_reward, compute_reward_with_overlong_penalty
  profiling_utils.py — NVTX ranges, ProfilerControl, MetricTracker, Timer
  sft_train.py     — SFT training loop (WORKS)
  rl_train.py      — Unified GRPO/Dr.GRPO/DAPO training (BROKEN)
  rl_train_overlapped.py — GRPO with CUDA stream overlap for rollout/update pipelining
  sweep.py         — Parameter sweep for finding optimal batch config
  baseline_eval.py — Evaluation script for GSM8K/MATH500
  eval.py          — Standalone eval script
scripts/           — Shell scripts for running/profiling each stage
```

## Key Technical Details

- **FSDP** shards across 2 GPUs. Each `Qwen2DecoderLayer` is an FSDP unit.
- **bf16 mixed precision** throughout. Activation checkpointing enabled.
- **Memory budget**: ~18GB/GPU for policy (weights + grads + Adam). ~6GB headroom.
- **No reference model** for Dr.GRPO and DAPO (saves 3GB/GPU). GRPO needs one (tight on 24GB).
- **SFT config that worked**: micro_batch=3, accum=4, seq_len=512, effective_batch=24, 764 tok/s, 23.5GB peak.

## RL Training Flow (what should happen in rl_train.py)

1. Load SFT checkpoint as policy model, wrap with FSDP
2. For each step:
   a. Get batch of prompts from GSM8K
   b. `generate_rollouts()` — generate G=4 completions per prompt using `model.module.generate()`
   c. `compute_rollout_rewards()` — binary exact-match reward (0 or 1)
   d. `rl_step()` — compute advantages, compute policy loss, backward
   e. Gradient clip + optimizer step
3. Log metrics, save checkpoints

## The Current Bug

The crash happens after "Memory before training" prints, before any training step logs appear. Likely in `generate_rollouts()` which calls `model.module.generate()` on an FSDP-wrapped model. FSDP + generate can be tricky — the model needs to be in eval mode and may need `FSDP.summon_full_params()` or similar for autoregressive generation.

## Commands

```bash
# SFT (works)
torchrun --nproc_per_node=2 src/sft_train.py --config configs/sft_config.yaml

# RL (broken)
torchrun --nproc_per_node=2 src/rl_train.py --algo dr_grpo --config configs/dr_grpo_config.yaml

# Eval
python src/baseline_eval.py --model checkpoints/sft/final/step_300 --dataset gsm8k --max_samples 50 --stage sft --prompt_mode sft --batch_size 8 --max_new_tokens 1024

# Environment
export HF_HOME=/workspace/hf_cache
export NCCL_IGNORE_DISABLED_P2P=1
```

## Known Issues & Fixes Applied

1. Base model generates hallucinated follow-up Q&A — fixed with stop string truncation
2. max_new_tokens was 1024, now 512 for SFT (but RL rollouts need 1024)
3. Pass@K had wrong estimator — fixed with unbiased formula
4. NCCL P2P disabled warnings — suppressed with NCCL_IGNORE_DISABLED_P2P=1
5. micro_batch=2 seq_len=1024 OOMs on 4090 — using micro_batch=3 seq_len=512
6. SFT loss plateau at 2.2 — likely due to seq_len=512 truncating long CoT traces