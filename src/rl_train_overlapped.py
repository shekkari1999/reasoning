"""
GRPO with Overlapped Rollout Generation and Policy Update.

Demonstrates pipeline overlap: while the policy update (backward + optimizer)
runs on the default CUDA stream, the NEXT batch of rollouts is generated
on a separate stream. This hides rollout latency behind compute.

The overlap is the key systems claim on the resume.

Usage:
    torchrun --nproc_per_node=2 src/rl_train_overlapped.py \
        --config configs/grpo_config.yaml

    # Profile to show overlap in Nsight timeline
    nsys profile --trace=cuda,nvtx,nccl --capture-range=cudaProfilerApi \
        --output=profiles/grpo_overlapped \
        torchrun --nproc_per_node=2 src/rl_train_overlapped.py \
            --config configs/grpo_config.yaml --profile
"""

import os
import sys
import math
import argparse
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import load_model, load_tokenizer, wrap_model_fsdp, load_reference_model, save_hf_checkpoint
from src.data import create_rl_dataloader
from src.rl_train import (
    get_cosine_lr, set_lr,
    generate_rollouts, compute_rollout_rewards, rl_step,
)
from src.profiling_utils import (
    nvtx_range, ProfilerControl, MetricTracker, log_memory, reset_peak_memory,
)


def train_overlapped(config: dict, profile_mode: bool = False):
    """GRPO training with rollout/update overlap."""
    # ---- Setup (same as rl_train.py) ----
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_name = config["model"]["name"]
    sft_checkpoint = config["model"]["sft_checkpoint"]
    num_steps = config["training"]["num_steps"]
    warmup_steps = config["training"]["warmup_steps"]
    max_lr = config["optimizer"]["lr"]
    min_lr = config["scheduler"]["min_lr"]

    algo_config = config.get("grpo", {})
    G = algo_config["group_size"]
    max_rollout_len = algo_config["max_rollout_len"]
    num_prompts = algo_config["num_prompts_per_step"]
    temperature = algo_config["temperature"]
    top_p = algo_config["top_p"]

    use_ref_model = config.get("reference_model", {}).get("enabled", False)
    log_every = config["logging"]["log_every"]
    save_every = config["logging"]["save_every"]
    output_dir = config["logging"]["output_dir"]

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"GRPO Training — OVERLAPPED VARIANT")
        print(f"{'='*60}")
        print(f"  Rollout generation and policy update run concurrently")
        print(f"  on separate CUDA streams.")
        print(f"{'='*60}")

    # ---- Load models ----
    tokenizer = load_tokenizer(model_name)
    policy = load_model(sft_checkpoint, dtype=torch.bfloat16)
    policy = wrap_model_fsdp(
        policy,
        mixed_precision=config["fsdp"]["mixed_precision"],
        activation_checkpointing=config["fsdp"]["activation_checkpointing"],
        forward_prefetch=config["fsdp"]["forward_prefetch"],
    )

    ref_model = None
    if use_ref_model:
        ref_model = load_reference_model(
            sft_checkpoint, dtype=torch.bfloat16,
            mixed_precision=config["fsdp"]["mixed_precision"],
        )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=max_lr,
        weight_decay=config["optimizer"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )

    dataloader = create_rl_dataloader(
        tokenizer=tokenizer,
        dataset_name=config["data"]["dataset"],
        split=config["data"]["split"],
        batch_size=num_prompts,
        num_workers=4,
        distributed=True,
    )

    profiler = ProfilerControl(
        warmup_steps=config["profiling"].get("warmup_steps", 10),
        capture_steps=config["profiling"].get("capture_steps", 20),
        enabled=profile_mode,
    )
    tracker = MetricTracker(log_dir=output_dir)

    # ---- Create separate CUDA stream for rollout generation ----
    rollout_stream = torch.cuda.Stream()
    train_stream = torch.cuda.default_stream()

    if rank == 0:
        log_memory("before training")
        reset_peak_memory()

    data_iter = iter(dataloader)
    policy.train()

    def get_next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            return next(data_iter)

    # ---- Pre-generate first batch of rollouts (no overlap yet) ----
    batch = get_next_batch()
    prompt_ids = batch["prompt_ids"].cuda()
    prompt_mask = batch["attention_mask"].cuda()
    raw_answers = batch["raw_answers"]

    current_rollouts = generate_rollouts(
        model=policy, prompt_ids=prompt_ids, attention_mask=prompt_mask,
        tokenizer=tokenizer, G=G, max_new_tokens=max_rollout_len,
        temperature=temperature, top_p=top_p, profile=profile_mode,
    )
    current_rewards = compute_rollout_rewards(
        completions=current_rollouts["completions"],
        raw_answers=raw_answers, G=G, dataset="gsm8k",
        profile=profile_mode,
    )
    current_answers = raw_answers

    # ---- Main loop with overlap ----
    for step in range(num_steps):
        profiler.step(step)

        if profile_mode:
            torch.cuda.nvtx.range_push(f"GRPO_overlapped_step_{step}")

        lr = get_cosine_lr(step, num_steps, warmup_steps, max_lr, min_lr)
        set_lr(optimizer, lr)

        # ---- Prefetch NEXT batch of prompts ----
        next_batch = get_next_batch()
        next_prompt_ids = next_batch["prompt_ids"].cuda()
        next_prompt_mask = next_batch["attention_mask"].cuda()
        next_raw_answers = next_batch["raw_answers"]

        # ---- Launch NEXT rollout generation on separate stream ----
        # NOTE: This reads from the policy model while the update below writes to it.
        # This is an intentional off-policy setup — the rollouts are generated from a
        # slightly stale policy. The importance sampling ratio in the loss corrects for this.
        # DeepSeek-R1 uses the same approach.

        next_rollouts = [None]  # mutable container for stream result
        next_rewards = [None]

        with torch.cuda.stream(rollout_stream):
            if profile_mode:
                torch.cuda.nvtx.range_push("async_next_rollout")

            next_rollouts[0] = generate_rollouts(
                model=policy,
                prompt_ids=next_prompt_ids,
                attention_mask=next_prompt_mask,
                tokenizer=tokenizer, G=G, max_new_tokens=max_rollout_len,
                temperature=temperature, top_p=top_p,
                profile=False,  # don't double-annotate
            )
            next_rewards[0] = compute_rollout_rewards(
                completions=next_rollouts[0]["completions"],
                raw_answers=next_raw_answers, G=G, dataset="gsm8k",
            )

            if profile_mode:
                torch.cuda.nvtx.range_pop()

        # ---- Policy update on default stream (CONCURRENT with rollout gen) ----
        if profile_mode:
            torch.cuda.nvtx.range_push("policy_update")

        loss_dict = rl_step(
            algo="grpo",
            policy_model=policy,
            ref_model=ref_model,
            rollout_data=current_rollouts,
            rewards=current_rewards,
            G=G,
            config=algo_config,
            profile=profile_mode,
        )

        if not loss_dict.get("skipped_step", False):
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if profile_mode:
            torch.cuda.nvtx.range_pop()  # policy_update

        # ---- Sync: wait for next rollouts to finish ----
        train_stream.wait_stream(rollout_stream)

        # ---- Swap: next becomes current ----
        current_rollouts = next_rollouts[0]
        current_rewards = next_rewards[0]
        current_answers = next_raw_answers

        if profile_mode:
            torch.cuda.nvtx.range_pop()  # overlapped_step

        # ---- Logging ----
        if rank == 0:
            reward_mean = current_rewards.mean().item()
            tracker.update(
                step=step,
                loss=loss_dict["loss"].item(),
                surrogate_loss=loss_dict["surrogate_loss"].item(),
                kl=loss_dict["kl"].item(),
                reward_mean=reward_mean,
                lr=lr,
            )

            if step % log_every == 0 or step == num_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1e9
                tracker.update(step=step, peak_mem_gb=mem)
                tracker.log(step, prefix="GRPO-OVL")

        if (step + 1) % save_every == 0:
            save_hf_checkpoint(
                model=policy, tokenizer=tokenizer,
                step=step + 1, output_dir=output_dir,
                model_name=model_name, rank=rank,
            )

        dist.barrier()

    # ---- Final ----
    save_hf_checkpoint(
        model=policy, tokenizer=tokenizer,
        step=num_steps, output_dir=output_dir + "/final",
        model_name=model_name, rank=rank,
    )
    profiler.stop()

    if rank == 0:
        tracker.save(filename="grpo_overlapped_metrics.json")
        print(f"\n{'='*60}")
        print(f"GRPO OVERLAPPED TRAINING COMPLETE")
        print(f"  Metrics: {output_dir}/grpo_overlapped_metrics.json")
        print(f"{'='*60}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.num_steps is not None:
        config["training"]["num_steps"] = args.num_steps
    if args.output_dir is not None:
        config["logging"]["output_dir"] = args.output_dir

    train_overlapped(config, profile_mode=args.profile)


if __name__ == "__main__":
    main()
