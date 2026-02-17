# Building a Reasoning LLM from Scratch

Implemented the complete post-training pipeline to turn a base LLM into a reasoning model. Covers inference-time scaling, self-refinement, and reinforcement learning with verifiable rewards (GRPO). No TRL, no alignment libraries. Inspired by **DeepSeek-R1**.

Evaluated on **GSM8K & MATH** benchmarks. Runs on single GPU.

## Training Pipeline

```
Base Model → Evaluate → CoT + Sampling → Self-Refine → GRPO Train → Reasoning LLM
```

## Components

### Inference Engine
Custom text generation with KV-cache, temperature/top-p sampling, and `torch.compile` optimization.

### Evaluation Harness
Math verifier with SymPy symbolic equivalence checking, structured answer extraction, and GSM8K/MATH benchmarking.

### Inference-Time Scaling
Chain-of-thought prompting, self-consistency via majority voting (N=10), and systematic temperature analysis.

### Self-Refinement
Iterative generate-score-critique loop using token log-probability confidence scoring.

### GRPO from Scratch
Full RL pipeline: rollout sampling, group-relative advantages, clipped policy gradient with KL penalty, multi-epoch training with checkpointing.

## Project Structure

```
├── src/
│   ├── inference.py          # KV-cache generation, temperature/top-p sampling
│   ├── grpo.py               # GRPO training loop from scratch
│   ├── self_refine.py        # Self-refinement with log-prob scoring
│   └── cot_sampling.py       # Chain-of-thought + self-consistency
├── eval/
│   ├── math_verifier.py      # SymPy symbolic equivalence checker
│   ├── answer_extraction.py  # Structured answer parsing
│   └── benchmark.py          # GSM8K/MATH evaluation runner
├── configs/
│   └── default.yaml          # Training and evaluation configs
├── notebooks/
│   └── exploration.ipynb     # Interactive experiments
├── scripts/
│   ├── train_sft.py          # SFT training script
│   ├── train_grpo.py         # GRPO training script
│   └── evaluate.py           # Run benchmarks
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/shekkari1999/reasoning.git
cd reasoning
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### 1. Evaluate base model
```bash
python scripts/evaluate.py --model Qwen/Qwen3-0.6B --benchmark gsm8k
```

### 2. Run SFT
```bash
python scripts/train_sft.py --model Qwen/Qwen3-0.6B --epochs 3
```

### 3. Inference-time scaling
```bash
python -m src.cot_sampling --model Qwen/Qwen3-0.6B --n_samples 10 --benchmark gsm8k
```

### 4. Train with GRPO
```bash
python scripts/train_grpo.py --model Qwen/Qwen3-0.6B --epochs 5 --batch_size 8
```

## Tech Stack

**Python** · **PyTorch** · **HuggingFace Transformers** · **SymPy** · **W&B** · **CUDA**
