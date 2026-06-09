# How to run (Reasoning_model)

Simple order: smoke test locally, then rent GPU for eval and training.

## 1. Local setup (Mac or GPU box)

```bash
cd Reasoning_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py
```

If smoke tests pass, the reward logic and GRPO loss math are wired correctly.

## 2. GPU to rent

| Job | GPU | Why |
|-----|-----|-----|
| Baseline eval + pass@k | **1x H100 80GB** | Qwen2.5-3B fits easily; batch 16 is fast |
| SFT + RL training | **2x H100 80GB** or **2x A100 40GB** | Scripts use `torchrun --nproc_per_node=2` + FSDP |

H100 is the best signal for your portfolio (matches triton-kernels / mini-vllm). A100 40GB works if H100 is booked.

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

# pass@k analysis (200 samples, sampled decoding)
python src/baseline_analysis.py \
  --model Qwen/Qwen2.5-3B \
  --pass_k_samples 200
```

Results land in `results/*_eval_summary.json`.

## 4. Training order (after baseline numbers exist)

```bash
# SFT first
bash scripts/run_sft.sh

# RL — start with Dr.GRPO (no ref model, easiest on memory)
bash scripts/run_dr_grpo.sh

# Then compare algorithms
bash scripts/run_grpo.sh
bash scripts/run_dapo.sh

# Re-eval all checkpoints
bash scripts/run_eval.sh
```

## 5. What to compare

Same benchmarks, same test split, same eval script:

```
Base Qwen2.5-3B  →  SFT  →  SFT + Dr.GRPO / GRPO / DAPO
         ↓              ↓              ↓
              GSM8K test + MATH500 (pass@1, pass@8)
```

Only claim numbers that appear in `results/*_eval_summary.json`.

## 6. If something breaks

Log it in `experiments/debug_log.md` with:
- command you ran
- error message
- what you tried next

That file is part of the deliverable, not optional.
