"""
Supervised Fine-Tuning on Chain-of-Thought Traces.

Trains Qwen-2.5-3B on CoT data to produce <think>...</think><answer>...</answer> format.
Pure PyTorch — no TRL, no HF Trainer.

Features:
  - FSDP sharding across GPUs
  - bf16 mixed precision
  - Gradient accumulation
  - Activation checkpointing
  - NVTX annotations for Nsight Systems profiling
  - Cosine LR schedule with warmup

Usage:
    # Training
    torchrun --nproc_per_node=2 src/sft_train.py --config configs/sft_config.yaml

    # Profile with Nsight Systems
    nsys profile --trace=cuda,nvtx,nccl --capture-range=cudaProfilerApi \
        --output=profiles/sft \
        torchrun --nproc_per_node=2 src/sft_train.py --config configs/sft_config.yaml --profile
"""

import os
import sys
import math
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import load_model, load_tokenizer, wrap_model_fsdp, save_hf_checkpoint
from src.data import create_sft_dataloader
from src.profiling_utils import (
    nvtx_range,
    ProfilerControl,
    MetricTracker,
    Timer,
    log_memory,
    reset_peak_memory,
)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_cosine_lr(step: int, total_steps: int, warmup_steps: int,
                  max_lr: float, min_lr: float) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: FSDP,
    batch: dict,
    accum_steps: int,
    vocab_size: int,
    step: int,
    profile: bool = False,
) -> dict:
    """Single training step with gradient accumulation.

    Args:
        model: FSDP-wrapped model
        batch: dict with input_ids, labels, attention_mask
        accum_steps: gradient accumulation steps (batch is already one micro-batch)
        vocab_size: for cross-entropy
        step: current step number (for NVTX)
        profile: whether to add NVTX annotations

    Returns:
        dict with loss value
    """
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # Forward
    if profile:
        torch.cuda.nvtx.range_push(f"forward_step{step}")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten and compute loss — only on non-masked tokens (labels != -100)
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        # Scale by accumulation steps
        loss = loss / accum_steps

    if profile:
        torch.cuda.nvtx.range_pop()

    # Backward
    if profile:
        torch.cuda.nvtx.range_push(f"backward_step{step}")

    loss.backward()

    if profile:
        torch.cuda.nvtx.range_pop()

    return {"loss": loss.detach() * accum_steps}  # return unscaled loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, profile_mode: bool = False):
    # ---- Distributed setup ----
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- Config ----
    model_name = config["model"]["name"]
    micro_batch_size = config["training"]["micro_batch_size"]
    accum_steps = config["training"]["gradient_accumulation_steps"]
    seq_len = config["training"]["seq_len"]
    num_steps = config["training"]["num_steps"]
    warmup_steps = config["training"]["warmup_steps"]

    max_lr = config["optimizer"]["lr"]
    min_lr = config["scheduler"]["min_lr"]
    weight_decay = config["optimizer"]["weight_decay"]
    betas = tuple(config["optimizer"]["betas"])
    eps = config["optimizer"]["eps"]

    log_every = config["logging"]["log_every"]
    save_every = config["logging"]["save_every"]
    output_dir = config["logging"]["output_dir"]

    effective_batch = micro_batch_size * accum_steps * world_size

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"SFT Training — {model_name}")
        print(f"{'='*60}")
        print(f"  GPUs: {world_size}")
        print(f"  micro_batch={micro_batch_size} × accum={accum_steps} × "
              f"world={world_size} = effective_batch={effective_batch}")
        print(f"  seq_len={seq_len} | steps={num_steps} | lr={max_lr}")
        print(f"  Profile mode: {profile_mode}")
        print(f"{'='*60}")

    # ---- Load model & tokenizer ----
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, dtype=torch.bfloat16)
    vocab_size = model.config.vocab_size

    # ---- FSDP wrap ----
    model = wrap_model_fsdp(
        model,
        mixed_precision=config["fsdp"]["mixed_precision"],
        activation_checkpointing=config["fsdp"]["activation_checkpointing"],
        forward_prefetch=config["fsdp"]["forward_prefetch"],
    )

    if rank == 0:
        log_memory("after model load")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    # ---- Dataloader ----
    dataloader = create_sft_dataloader(
        tokenizer=tokenizer,
        dataset_name=config["data"]["dataset"],
        subset=config["data"].get("subset", "default"),
        split=config["data"]["split"],
        max_samples=config["data"].get("max_samples", 10000),
        max_seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        num_workers=4,
        distributed=True,
    )

    # ---- Profiling setup ----
    profiler = ProfilerControl(
        warmup_steps=config["profiling"].get("warmup_steps", 10),
        capture_steps=config["profiling"].get("capture_steps", 20),
        enabled=profile_mode,
    )
    tracker = MetricTracker(log_dir=output_dir)
    timer = Timer(cuda_sync=True)

    if rank == 0:
        log_memory("before training")
        reset_peak_memory()

    # ---- Training loop ----
    data_iter = iter(dataloader)
    model.train()

    for step in range(num_steps):
        profiler.step(step)

        if profile_mode:
            torch.cuda.nvtx.range_push(f"SFT_step_{step}")

        # Cosine LR with warmup
        lr = get_cosine_lr(step, num_steps, warmup_steps, max_lr, min_lr)
        set_lr(optimizer, lr)

        step_loss = 0.0

        # ---- Gradient accumulation ----
        for micro_step in range(accum_steps):
            # Get next batch (cycle through dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            result = train_step(
                model=model,
                batch=batch,
                accum_steps=accum_steps,
                vocab_size=vocab_size,
                step=step,
                profile=profile_mode,
            )
            step_loss += result["loss"].item()

        # ---- Optimizer step ----
        if profile_mode:
            torch.cuda.nvtx.range_push("optimizer_step")

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if profile_mode:
            torch.cuda.nvtx.range_pop()  # optimizer_step
            torch.cuda.nvtx.range_pop()  # SFT_step

        # ---- Logging ----
        if rank == 0:
            tracker.update(
                step=step,
                loss=step_loss,
                lr=lr,
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            )

            if step % log_every == 0 or step == num_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1e9
                tokens_seen = (step + 1) * effective_batch * seq_len
                tracker.update(step=step, peak_mem_gb=mem, tokens_seen=tokens_seen)
                tracker.log(step, prefix="SFT")

        # ---- Checkpointing ----
        if (step + 1) % save_every == 0 or step == num_steps - 1:
            if rank == 0:
                print(f"\n[Step {step}] Saving checkpoint...")

            save_hf_checkpoint(
                model=model,
                tokenizer=tokenizer,
                step=step + 1,
                output_dir=output_dir,
                model_name=model_name,
                rank=rank,
            )

    # ---- Final save ----
    if rank == 0:
        print(f"\n[Step {num_steps}] Saving final checkpoint...")

    save_hf_checkpoint(
        model=model,
        tokenizer=tokenizer,
        step=num_steps,
        output_dir=output_dir + "/final",
        model_name=model_name,
        rank=rank,
    )

    # ---- Cleanup ----
    profiler.stop()

    if rank == 0:
        tracker.save(filename="sft_metrics.json")
        log_memory("end of training")

        print(f"\n{'='*60}")
        print(f"SFT TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Steps: {num_steps}")
        print(f"  Final loss: {step_loss:.4f}")
        print(f"  Checkpoint: {output_dir}/final")
        print(f"  Metrics: {output_dir}/sft_metrics.json")
        print(f"{'='*60}")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--profile", action="store_true",
                        help="Enable NVTX profiling (use with nsys)")

    # Override config values from CLI
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    if args.micro_batch is not None:
        config["training"]["micro_batch_size"] = args.micro_batch
    if args.accum_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.accum_steps
    if args.seq_len is not None:
        config["training"]["seq_len"] = args.seq_len
    if args.num_steps is not None:
        config["training"]["num_steps"] = args.num_steps
    if args.output_dir is not None:
        config["logging"]["output_dir"] = args.output_dir

    train(config, profile_mode=args.profile)


if __name__ == "__main__":
    main()
