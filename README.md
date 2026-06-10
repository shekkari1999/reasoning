# Reasoning Engine

Post-training **Qwen2.5-3B** into a reasoning model using raw PyTorch — a lab-grade reproduction of the **DeepSeek-R1 pipeline**: SFT on chain-of-thought traces, then **RLVR** (reinforcement learning with verifiable rewards) via Group Relative Policy Optimization.

No TRL. No HuggingFace Trainer. Every training loop, loss function, and FSDP strategy is implemented from scratch.

**Repo:** https://github.com/shekkari1999/reasoning

---

## The Story (What This Project Is For)

Base models like Qwen2.5-3B already *can* solve many math problems — but only if you sample enough times. Our baseline pass@k analysis shows a **~29% gap** between pass@1 and pass@8 on both GSM8K and MATH500: the model has the capability, but single-shot decoding doesn't surface it.

This project asks: **can a minimal DeepSeek-R1-style pipeline close that gap?**

```
Base Qwen2.5-3B  →  SFT (CoT traces)  →  Dr.GRPO (verifiable rewards)
       ↓                    ↓                        ↓
   GSM8K + MATH500     same benchmarks, same eval harness
   pass@1 + pass@8
```

**What makes this portfolio-grade (not a tutorial fork):**
- Three RL algorithms implemented and comparable: **GRPO**, **Dr.GRPO**, **DAPO**
- **FSDP** training on 2 GPUs with memory profiling and parameter sweeps
- **Honest benchmarking** — same test splits, documented prompt modes, no metric games
- **Training dynamics debugging** — reward curves, loss spikes, OOM post-mortems documented

---

## What We've Done So Far (~$50 on H100)

### Phase 1: Baseline + Pass@k (DONE ✅)

**Hardware:** 1× H100 80GB | **Cost:** ~$10–15 | **Time:** ~2.5 hours

| Benchmark | Accuracy | Correct/Total | Decode |
|-----------|----------|---------------|--------|
| **GSM8K** | **71.4%** | 942/1319 | greedy, few-shot |
| **MATH500** | **37.6%** | 188/500 | greedy, few-shot |

**Pass@k (200 GSM8K + 150 MATH500, T=0.7, K=8):**

| Metric | GSM8K | MATH500 |
|--------|-------|---------|
| pass@1 | 65.9% | 35.1% |
| pass@8 | **94.5%** | **64.0%** |
| **Gap (pass@8 − pass@1)** | **28.6%** | **28.9%** |

**64% of GSM8K problems are "mixed"** (correct in some samples, wrong in others) — the regime GRPO is designed for.

Results committed: `results/base_eval_summary.json`, `results/base_extended_analysis.json`, plots in `results/`.

---

### Phase 2: First SFT + Dr.GRPO Run (DONE — flawed config ⚠️)

**Hardware:** 2× H100 80GB | **Cost:** ~$30–40 | **Time:** ~4 hours training + eval

#### SFT v1 (seq_len=512, 300 steps, 0.72 epochs)

| Setting | Value |
|---------|-------|
| Data | OpenR1-Math-220k, 10k subset |
| seq_len | **512** (truncated long CoT traces) |
| Effective batch | 24 (mb=3 × accum=4 × 2 GPU) |
| Peak VRAM | 16.6 GB/GPU |
| Training time | ~1.3 hours |

| Benchmark | Accuracy | vs Base (fair prompt) |
|-----------|----------|----------------------|
| GSM8K (`prompt_mode=sft`) | **30.1%** | base+sft-prompt not yet run |
| MATH500 (`prompt_mode=sft`) | **11.2%** | — |

**Diagnosis:** SFT learned the output *format* (`<think>…<answer>…`) but not full reasoning — loss plateaued at ~2.2 because **512 tokens cut off most OpenR1 traces**. Training on OpenR1-only also caused **domain shift** away from GSM8K-style problems.

#### Dr.GRPO v1 (200 steps, max_rollout_len=1024)

| Setting | Value |
|---------|-------|
| Init from | `checkpoints/sft/final/step_300` |
| Data | GSM8K train |
| Peak VRAM | 50.6 GB/GPU |
| reward_mean | ~0.5–0.56 (bounced 0.25–0.75) |

| Benchmark | Accuracy (est.) |
|-----------|-----------------|
| GSM8K | **~34%** (modest +4% over SFT) |

Directionally correct — RL helped slightly — but built on a weak SFT foundation.

---

## What Went Wrong in v1 (Lessons)

| Issue | Impact | Fix in v2 |
|-------|--------|-----------|
| `seq_len=512` | Truncated CoT; loss plateau | **`seq_len=4096`** |
| 300 steps (0.72 epoch) | Under-trained | **625 steps (1 full epoch)** |
| OpenR1 only | GSM8K catastrophic forgetting | *Optional:* add GSM8K to SFT mix |
| Eval `max_new_tokens=512` | Answers cut before `</answer>` | **`max_new_tokens=2048`** at eval |
| Base few-shot vs SFT zero-shot | Unfair headline comparison | Run **base + `prompt_mode=sft`** floor |
| 40 GB disk | Checkpoint OOM at step 100 | **150–200 GB EBS** on AWS |

**What was already fine:** FSDP + activation checkpointing + bf16, cosine LR, gradient accumulation, loss masking on prompts, reward function, GRPO loss implementation.

---

## The $250 Run Plan (AWS g7e.12xlarge)

**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (**96 GB/GPU**)  
**AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Amazon Linux 2023, **64-bit x86**)  
**Instance:** `g7e.12xlarge` (~$8.29/hr on-demand)  
**Storage:** 200 GB gp3 root volume (checkpoints are large)

### Budget Breakdown

| Phase | Steps / Work | Est. Time | Est. Cost |
|-------|--------------|-----------|-----------|
| **0. Setup + memory probe** | 5-step sweep @ seq=4096 | 30 min | $4 |
| **1. Fair base eval** | base + `prompt_mode=sft`, 2048 tokens | 1.5 hr | $12 |
| **2. SFT v2** | 625 steps, seq=4096, 1 epoch | **~12–15 hr** | **~$100–125** |
| **3. SFT eval** | GSM8K + MATH500 | 1 hr | $8 |
| **4. Dr.GRPO v2** | 200–400 steps on new SFT | 3–6 hr | $25–50 |
| **5. Final eval + pass@k** | all checkpoints | 2 hr | $17 |

**$250 covers Phase 0–3 (SFT v2 + evals).** Dr.GRPO v2 adds ~$30–50 if budget allows.

> **Tip:** Use `tmux` for all long runs. Stop the instance when idle — EBS persists, compute does not.

---

## SFT v2 Config (current)

```yaml
# configs/sft_config.yaml
micro_batch_size: 1              # required at long seq_len (~45-50 GB/GPU at 4096)
gradient_accumulation_steps: 8
seq_len: 4096                    # full CoT for GSM8K + most MATH500 (was 512)
num_steps: 625                   # 1 epoch: 625 × 16 = 10,000 samples
warmup_steps: 63
save_every: 200
```

**Predicted peak VRAM:** ~45–50 GB/GPU (comfortable on 96 GB)  
**Predicted training time:** ~12–15 hours for 625 steps

---

## Run Order on AWS

### 0. Launch + setup

```bash
# SSH into g7e.12xlarge
git clone https://github.com/shekkari1999/reasoning.git && cd reasoning
pip install -r requirements.txt

# Sanity check
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

### 1. Memory probe (do NOT skip)

```bash
tmux new -s probe
torchrun --nproc_per_node=2 src/sweep.py \
  --mode single --micro_batch 1 --accum_steps 8 --seq_len 4096 --num_steps 5
# If peak < 90 GB → proceed.
```

### 2. Fair baseline floor

```bash
python src/baseline_eval.py \
  --model Qwen/Qwen2.5-3B \
  --dataset both \
  --stage base_sft_prompt \
  --prompt_mode sft \
  --max_new_tokens 2048 \
  --batch_size 16
```

### 3. SFT v2 training (~12–15 hours)

```bash
tmux new -s sft
bash scripts/run_sft.sh
# Checkpoints: checkpoints/sft/ at steps 200, 400, 600, 625
```

### 4. Evaluate SFT v2

```bash
python src/baseline_eval.py \
  --model checkpoints/sft/final/step_625 \
  --dataset both \
  --stage sft_v2 \
  --prompt_mode sft \
  --max_new_tokens 2048 \
  --batch_size 16
```

### 5. Dr.GRPO v2 (if budget remains)

Update `configs/dr_grpo_config.yaml`:
```yaml
sft_checkpoint: "checkpoints/sft/final/step_625"
```

```bash
tmux new -s rl
bash scripts/run_dr_grpo.sh
```

### 6. Final ladder eval

```bash
# Evaluate all checkpoints with identical settings
for ckpt in base sft_v2 dr_grpo_v2; do
  python src/baseline_eval.py --model $CKPT_PATH --dataset both \
    --prompt_mode sft --max_new_tokens 2048 --batch_size 16
done

python src/baseline_analysis.py \
  --model checkpoints/dr_grpo/final/step_200 \
  --pass_k_samples 200
```

---

## Results Ladder (target state after $250 run)

| Stage | GSM8K | MATH500 | Peak Mem/GPU | Prompt Mode | Status |
|-------|-------|---------|--------------|-------------|--------|
| Base (few-shot) | **71.4%** | **37.6%** | — | `base` | ✅ Done |
| Base (sft-prompt floor) | — | — | — | `sft` | 🔲 This run |
| SFT v1 (seq=512) | 30.1% | 11.2% | 16.6 GB | `sft` | ✅ Done (flawed) |
| **SFT v2 (seq=4096)** | **—** | **—** | ~49 GB | `sft` | 🔲 This run |
| Dr.GRPO v1 | ~34% | — | 50.6 GB | `sft` | ✅ Done (on weak SFT) |
| **Dr.GRPO v2** | **—** | **—** | ~50 GB | `sft` | 🔲 If budget allows |

**Success criteria for SFT v2:**
- GSM8K > 30% (beat v1) with fair prompt
- MATH500 > 11% (beat v1)
- Loss continues decreasing past step 300 (no plateau at 2.2)
- Model produces complete `</answer>` tags at 2048 decode length

**Success criteria for Dr.GRPO v2:**
- GSM8K > SFT v2 by ≥3–5% (RL adds value beyond SFT)
- pass@8 − pass@1 gap narrows vs base (RL surfaces latent capability)

---

## What This Project Does (Technical)

### Stage 1 — Supervised Fine-Tuning

Train on chain-of-thought traces from OpenR1-Math-220k. Target format:

```
<think>
{step-by-step reasoning}
</think>
<answer>{final answer}</answer>
```

Loss computed only on completion tokens (prompt masked with `labels=-100`).

### Stage 2 — Reinforcement Learning (RLVR)

Three algorithms implemented as an ablation:

| Algorithm | KL Penalty | Reference Model | Key Difference |
|-----------|-----------|----------------|--------------|
| **GRPO** | Yes (β=0.1) | Yes (frozen) | Vanilla DeepSeek-R1 |
| **Dr.GRPO** | No | No | Drops KL + ref model, saves ~3 GB/GPU |
| **DAPO** | No | No | Asymmetric clip + dynamic sampling |

All use **group-relative advantages** — G completions per prompt serve as their own baseline, no learned critic.

**Reward:** binary exact-match on extracted answer. No learned reward model.

---

## Architecture & Systems

### Distributed Training

- **FSDP FULL_SHARD** across 2 GPUs — weights, gradients, optimizer states sharded per transformer block
- **bf16** mixed precision with activation checkpointing
- **Gradient accumulation** for effective batch size 16–24 within VRAM limits

### Memory (measured)

| Stage | Config | Peak VRAM/GPU |
|-------|--------|---------------|
| SFT v1 | mb=3, seq=512 | 16.6 GB |
| SFT v2 (predicted) | mb=1, seq=4096 | ~45–50 GB |
| Dr.GRPO v1 | G=4, rollout=1024 | 50.6 GB |

Linear model from sweep data: `peak ≈ 20 GB + 0.007 × (micro_batch × seq_len)`

### Profiling

NVTX-annotated training loops for Nsight Systems. Parameter sweep (`src/sweep.py`) finds optimal batch config per hardware.

---

## Project Structure

```
reasoning/
├── configs/
│   ├── sft_config.yaml          # SFT v2: seq=4096, 625 steps
│   ├── dr_grpo_config.yaml      # RL on GSM8K
│   ├── grpo_config.yaml         # RL with KL + ref model
│   └── dapo_config.yaml         # Asymmetric clip variant
├── src/
│   ├── sft_train.py             # SFT training loop
│   ├── rl_train.py              # GRPO / Dr.GRPO / DAPO
│   ├── losses.py                # Policy gradient losses
│   ├── rewards.py               # Answer extraction + binary reward
│   ├── baseline_eval.py         # GSM8K + MATH500 eval
│   ├── baseline_analysis.py     # pass@k, consistency, plots
│   └── sweep.py                 # Memory + throughput sweep
├── scripts/
│   ├── run_sft.sh
│   ├── run_dr_grpo.sh
│   └── run_memory_analysis.sh
├── results/                     # JSON metrics, plots (committed)
└── checkpoints/                 # gitignored — save to EBS / HF Hub
```

---

## GRPO Algorithm (summary)

1. **Rollout:** sample G completions per prompt (temperature=0.7)
2. **Reward:** exact-match on extracted answer
3. **Advantage:** `A_i = (r_i − mean(r)) / (std(r) + ε)` (Dr.GRPO drops std norm)
4. **Loss:** clipped surrogate `L = −min(ratio·A, clip(ratio)·A)`
5. **KL** (GRPO only): `L + β·KL(π_policy || π_ref)`

---

## Reproducing Locally (CPU smoke test)

```bash
pip install -r requirements.txt
python tests/smoke_test.py   # rewards + loss math, no GPU
```

Full SFT v2 requires 2× GPU with ≥50 GB/GPU (g7e.12xlarge recommended).

---

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — reasoning via RL
- [Dr. GRPO](https://arxiv.org/abs/2503.02846) — no reference model
- [DAPO](https://arxiv.org/abs/2503.14476) — asymmetric clipping
- [GRPO](https://arxiv.org/abs/2402.03300) — group relative policy optimization
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) — SFT data

---

## Interview Talking Points

1. **"Why RL?"** — pass@8 is 29% higher than pass@1; model has latent capability RL can surface
2. **"Why did v1 SFT fail?"** — seq_len=512 truncated CoT; diagnosed via loss plateau + ablation
3. **"Why 4096?"** — covers 98%+ GSM8K traces and most MATH500; 512 was the real bug
4. **"What's fair comparison?"** — same benchmark, same prompt mode, same decode length
5. **"What did you implement?"** — FSDP training, 3 RL variants, reward function, eval harness — not a TRL wrapper
