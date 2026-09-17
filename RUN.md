# How to run (Reasoning_model)

Simple order: smoke test locally, then rent GPU for eval and training.

## 1. Local setup (Mac or GPU box)

```bash
cd reasoning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py
```

The smoke tests exercise the production Dr.GRPO objective, completion masking,
deterministic sampling, reward logic, and result aggregation.

## 2. GPU to rent

| Job | GPU | Why |
|-----|-----|-----|
| Baseline eval | **1x H100 80GB** | Qwen2.5-3B fits easily; batch 16 is fast |
| SFT + RL training | **2x 96GB Blackwell**, **2x H100 80GB**, or **2x A100 40GB** | Scripts use `torchrun --nproc_per_node=2` + FSDP |

Provider: RunPod or Lambda, Ubuntu 22.04, CUDA 12.x.

## 3. On the GPU instance

```bash
git clone <your-repo-url> reasoning_model && cd reasoning_model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py

# Quick sanity eval (20 problems, ~2 min)
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset gsm8k \
  --max_samples 20 \
  --stage base

# Full baseline (GSM8K + MATH500 test splits, ~30-60 min)
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset both \
  --stage base \
  --batch_size 16
```

New evaluations land in `results/*_eval_summary.json`. The headline run metrics
used by the README and resume live in `results/latest_run_summary.json`.

## 4. Training order (after baseline numbers exist)

```bash
# SFT first
bash scripts/run_sft.sh

# RL — Dr.GRPO (no ref model, easiest on memory)
bash scripts/run_dr_grpo.sh

# Re-eval all checkpoints
bash scripts/run_eval.sh
```

## 5. What to compare

Use the same benchmarks, test splits, prompt mode, and decoder settings for all
three stages:

```
Base Qwen2.5-3B  →  SFT  →  SFT + Dr.GRPO
         ↓              ↓              ↓
              GSM8K test + MATH500
```

`scripts/run_eval.sh` validates all three stage summaries and writes
`results/evaluator_run_summary.json` with detailed-artifact hashes.

## 6. Memory analysis

**Eval (1 GPU)** — memory is saved automatically in `results/*_eval_summary.json`:

```bash
bash scripts/run_memory_analysis.sh
# eval VRAM probe: results/memory_probe_eval_summary.json → memory.peak_gb_max
# (does not overwrite results/base_eval_summary.json)
```

**Pre-training sweep (2 GPU)** — estimate SFT vs RL headroom before a long run:

```bash
bash scripts/run_memory_analysis.sh
# or manually:
torchrun --nproc_per_node=2 src/sweep.py --mode quick --num_steps 10   # SFT
```

**During training** — peak memory per GPU is logged every `log_every` steps and saved at end:

| Stage | Live logs | Final report |
|-------|-----------|--------------|
| SFT | `peak_mem_gb` in stdout | `results/memory_sft.json` |
| Dr.GRPO | `mem=XX.XG` in tqdm | `results/memory_dr_grpo.json` |

Training metrics with per-step peaks: `checkpoints/sft/sft_metrics.json` and
`checkpoints/dr_grpo/dr_grpo_metrics.json`.

## 7. Optional Nsight profile

With Nsight Systems installed, capture the steady-state SFT window after ten
warmup steps:

```bash
bash scripts/profile_sft.sh
```

The report is written under `profiles/`, which is excluded from Git.
