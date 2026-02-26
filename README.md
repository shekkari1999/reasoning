# Reasoning Engine

Post-training Qwen-2.5-3B into a reasoning model using raw PyTorch — reproducing a DeepSeek-R1-style pipeline: supervised fine-tuning on chain-of-thought traces followed by reinforcement learning to elicit long-horizon reasoning without a learned critic.

No TRL. No HuggingFace Trainer. Every training loop, loss function, and distributed strategy is implemented from scratch.

## What This Project Does

Takes a base language model (Qwen-2.5-3B) and teaches it to reason step-by-step through math problems via two stages:

**Stage 1 — Supervised Fine-Tuning (SFT):** Train the model on chain-of-thought traces so it learns to produce `<think>...</think><answer>...</answer>` formatted reasoning.

**Stage 2 — Reinforcement Learning:** Further improve reasoning quality using reward signals from correct/incorrect answers. Three RL algorithms are implemented and compared as an ablation study:

| Algorithm | KL Penalty | Reference Model | Clipping | Key Difference |
|-----------|-----------|----------------|----------|----------------|
| **GRPO** | Yes (β=0.1) | Yes (frozen) | Symmetric ε=0.2 | Vanilla DeepSeek-R1 approach |
| **Dr. GRPO** | No | No | Symmetric ε=0.2 | Drops KL + ref model, saves 3GB/GPU |
| **DAPO** | No | No | Asymmetric ε_low=0.2, ε_high=0.28 | Dynamic sampling + overlong penalty |

All three use Group Relative Policy Optimization — the group of G completions per prompt serves as its own baseline, eliminating the need for a learned value function (critic).

## Key Results

### Baseline (Qwen-2.5-3B, no fine-tuning)

| Benchmark | Accuracy |
|-----------|----------|
| GSM8K | 70.0% |
| MATH500 | ~40% |

### Pass@K Analysis — Motivation for RL

| Metric | GSM8K |
|--------|-------|
| pass@1 (greedy) | 70.0% |
| pass@1 (sampled, T=0.7) | 67.7% |
| pass@8 (sampled, T=0.7) | 95.0% |
| **Gap (RL opportunity)** | **27.3%** |

The 27.3% gap between pass@1 and pass@8 demonstrates that the base model has significant latent reasoning capability that greedy decoding fails to surface. GRPO exploits this gap by sampling multiple completions, identifying which ones succeed, and reinforcing those trajectories.

### Training Pipeline Results

| Stage | GSM8K | MATH500 | Peak Mem/GPU |
|-------|-------|---------|--------------|
| Base | 70.0% | ~40% | — |
| SFT | — | — | — |
| GRPO | — | — | — |
| Dr. GRPO | — | — | — |
| DAPO | — | — | — |

*Results updated after training runs complete.*

## Architecture & Systems

### Distributed Training

- **FSDP (Fully Sharded Data Parallelism):** Model weights, gradients, and optimizer states sharded across 2 GPUs. Each transformer block (`Qwen2DecoderLayer`) is an independent FSDP unit.
- **Mixed Precision:** bf16 for forward/backward, bf16 for FSDP reduce operations. Tensor core utilization on Ada Lovelace (RTX 4090) / Ampere (A100).
- **Gradient Accumulation:** Multiple micro-batches accumulated before optimizer step, simulating larger effective batch sizes within memory constraints.
- **Activation Checkpointing:** Recomputes activations during backward pass instead of storing them. Trades compute for memory — essential for fitting 3B model training on 24GB GPUs.

### Memory Budget (2×RTX 4090 24GB)

```
POLICY (FSDP-sharded):
  Weights bf16:      3GB/GPU
  Gradients bf16:    3GB/GPU
  Adam fp32 (m+v):  12GB/GPU
  Subtotal:         18GB/GPU

Activations (checkpointed, micro_batch=1): ~2-3GB/GPU
CUDA overhead + rollout buffers:           ~2-3GB/GPU
Total:                                    ~22-24GB/GPU
```

Dr. GRPO and DAPO drop the reference model, saving 3GB/GPU — the difference between fitting and OOM on 24GB hardware.

### Rollout/Update Overlap

The overlapped variant (`rl_train_overlapped.py`) pipelines rollout generation with policy updates using separate CUDA streams:

```
Step N:   [====== policy update ======]
Step N+1:    [====== rollout generation (async) ======]
                                          ↑ runs concurrently on separate stream
```

Rollouts for step N+1 are generated from a slightly stale policy (before the step N update lands). The importance sampling ratio in the clipped surrogate loss corrects for this off-policy data. This is the same approach used in DeepSeek-R1's training infrastructure.

### Profiling

Every training phase is annotated with NVTX ranges for Nsight Systems timeline visualization:

- `forward`, `backward`, `optimizer_step` — SFT phases
- `rollout_generation`, `reward_computation`, `advantage_computation`, `policy_loss_backward` — GRPO phases
- `async_next_rollout`, `policy_update` — overlap phases

Nsight Compute kernel analysis measures tensor core utilization, memory throughput, and occupancy for the dominant GEMMs.

## Project Structure

```
reasoning-engine/
├── configs/
│   ├── sft_config.yaml              # SFT hyperparameters
│   ├── grpo_config.yaml             # GRPO with KL penalty
│   ├── dr_grpo_config.yaml          # Dr. GRPO — no KL, no ref model
│   └── dapo_config.yaml             # DAPO — asymmetric clip, dynamic sampling
│
├── src/
│   ├── model.py                     # FSDP wrapping, checkpoint save/load
│   ├── data.py                      # SFT dataset + RL prompt dataset
│   ├── losses.py                    # GRPO / Dr.GRPO / DAPO loss functions
│   ├── rewards.py                   # Answer extraction + reward computation
│   ├── profiling_utils.py           # NVTX, memory logging, metric tracker
│   ├── sweep.py                     # Parameter sweep + GRPO memory test
│   ├── sft_train.py                 # SFT training loop
│   ├── rl_train.py                  # Unified GRPO / Dr.GRPO / DAPO training
│   ├── rl_train_overlapped.py       # GRPO with CUDA stream overlap
│   ├── baseline_eval.py             # Evaluation on GSM8K / MATH500
│   ├── baseline_analysis.py         # Pass@K, length analysis, plots
│   └── eval.py                      # Standalone eval script
│
├── scripts/
│   ├── run_sweep.sh                 # Find optimal batch config
│   ├── run_sft.sh                   # SFT training
│   ├── run_grpo.sh                  # GRPO training
│   ├── run_dr_grpo.sh               # Dr. GRPO training
│   ├── run_dapo.sh                  # DAPO training
│   ├── run_grpo_overlapped.sh       # Overlapped variant
│   ├── run_eval.sh                  # Evaluate all checkpoints
│   ├── profile_sft.sh              # Nsight Systems — SFT
│   ├── profile_grpo.sh             # Nsight Systems — baseline vs overlapped
│   ├── profile_ncu.sh              # Nsight Compute — kernel analysis
│   └── profile_all_variants.sh     # Nsight Systems — all 3 RL variants
│
├── notebooks/
│   ├── 01_baseline_eval.ipynb       # Baseline eval (Colab)
│   └── 02_baseline_analysis.ipynb   # Extended analysis (Colab)
│
├── profiles/                        # .nsys-rep and .ncu-rep artifacts
├── results/                         # Plots, JSON metrics, eval results
└── data/                            # Dataset cache
```

## GRPO Algorithm

Group Relative Policy Optimization generates G completions per prompt, scores them, and uses the group as its own baseline:

**1. Rollout Generation:** Sample G completions per prompt from the current policy with temperature sampling.

**2. Reward Computation:** Binary exact-match reward — extract the answer, compare to ground truth. No learned reward model.

**3. Group-Relative Advantage:**
```
A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + ε)
```

**4. Clipped Surrogate Loss:**
```
ratio = exp(log π_new(a|s) - log π_old(a|s))
L = -min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
```

**5. KL Penalty (GRPO only):**
```
L_total = L_surrogate + β × KL(π_policy || π_reference)
```

### Algorithm Variants

**Dr. GRPO** removes the KL penalty and reference model entirely, relying solely on clipping to prevent policy collapse. Uses simplified advantages: `A_i = r_i - mean(r)` without standard deviation normalization.

**DAPO** introduces three modifications:
- **Asymmetric clipping:** Looser clip (ε=0.28) for positive advantages, tighter (ε=0.2) for negative — lets the policy explore more aggressively when it finds good solutions.
- **Dynamic sampling:** Skips prompt groups where all G completions received the same reward (zero gradient signal, wasted compute).
- **Overlong penalty:** Applies a negative reward (-0.5) when completions hit max length without producing a valid answer, discouraging rambling.

## Hardware

| Component | Spec |
|-----------|------|
| GPUs | 2× NVIDIA RTX 4090 (24GB VRAM each) |
| Interconnect | PCIe (no NVLink) |
| Precision | bf16 (Ada Lovelace tensor cores) |
| Profiling | Nsight Systems 2023.x, Nsight Compute 2023.x |

## Reproducing

### Prerequisites

```bash
pip install torch transformers datasets accelerate pyyaml matplotlib seaborn
```

### Run Order

```bash
# 1. Parameter sweep — find optimal batch config for your hardware
bash scripts/run_sweep.sh

# 2. Update configs with sweep results, then train SFT
bash scripts/run_sft.sh

# 3. RL training (pick one or all)
bash scripts/run_dr_grpo.sh    # recommended first — no ref model
bash scripts/run_dapo.sh
bash scripts/run_grpo.sh       # needs ref model — tight on 24GB

# 4. Overlapped variant (for profiling comparison)
bash scripts/run_grpo_overlapped.sh

# 5. Profile
bash scripts/profile_sft.sh
bash scripts/profile_grpo.sh
bash scripts/profile_ncu.sh

# 6. Evaluate all checkpoints
bash scripts/run_eval.sh
```

### Baseline Evaluation (Colab — free)

```bash
python src/baseline_eval.py --model Qwen/Qwen2.5-3B --dataset both
python src/baseline_analysis.py --model Qwen/Qwen2.5-3B --pass_k_samples 200
```

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [DAPO: An Open-Source LLM Reinforcement Learning System](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO: Removing the Reference Model from GRPO](https://arxiv.org/abs/2503.02846)
- [Building a Reasoning Model from Scratch — Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
