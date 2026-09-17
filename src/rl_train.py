"""
Reinforcement Learning Training — Dr.GRPO

Training loop with rollout generation, reward computation, and FSDP
infrastructure. Loss function and advantage computation live in losses.py.

Usage:
    torchrun --nproc_per_node=2 src/rl_train.py --config configs/dr_grpo_config.yaml
"""

import os
import sys
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import (
    load_model, load_tokenizer, wrap_model_fsdp,
    save_hf_checkpoint,
)
from src.data import create_rl_dataloader
from src.losses import (
    get_per_token_logprobs,
    sequence_logprobs,
    compute_advantages_dr_grpo,
    dr_grpo_loss,
)
from src.rewards import compute_reward
from src.profiling_utils import (
    ProfilerControl,
    MetricTracker,
    log_memory,
    reset_peak_memory,
    gather_all_gpu_memory,
    save_memory_report,
)


# ---------------------------------------------------------------------------
# LR schedule (same as SFT)
# ---------------------------------------------------------------------------

def get_cosine_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# FSDP-compatible generation
# ---------------------------------------------------------------------------

def _top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling. Returns (batch, 1) token indices."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Keep at least one token: mask tokens whose *preceding* cumulative prob >= top_p
    mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
    sorted_logits[mask] = float('-inf')
    logits = torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def generate_with_fsdp(
    model: FSDP,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Autoregressive generation that works with FSDP.

    Calls model() directly (through FSDP's forward) so FSDP handles
    parameter all-gather/reshard correctly on each forward pass.
    Uses KV cache for efficiency.

    Returns:
        generated: (batch, prompt_len + generated_len) full token IDs
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

    generated = input_ids
    cur_mask = attention_mask.clone()
    past_key_values = None

    for _ in range(max_new_tokens):
        if past_key_values is None:
            outputs = model(input_ids=generated, attention_mask=cur_mask, use_cache=True)
        else:
            outputs = model(input_ids=next_token, attention_mask=cur_mask,
                            past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :] / temperature

        next_token = _top_p_sample(logits, top_p)  # (batch, 1)

        # Pad finished sequences
        next_token = torch.where(
            unfinished.unsqueeze(-1), next_token,
            torch.full_like(next_token, pad_token_id),
        )

        generated = torch.cat([generated, next_token], dim=-1)
        cur_mask = torch.cat([cur_mask, unfinished.unsqueeze(-1).long()], dim=-1)

        # Check EOS
        unfinished = unfinished & (next_token.squeeze(-1) != eos_token_id)
        if not unfinished.any():
            break

    del past_key_values
    torch.cuda.empty_cache()
    return generated


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_rollouts(
    model: FSDP,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    G: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    profile: bool = False,
) -> dict:
    """Generate G completions per prompt using the current policy.

    Args:
        model: FSDP-wrapped policy
        prompt_ids: (batch, prompt_len) — left-padded prompt token IDs
        attention_mask: (batch, prompt_len)
        tokenizer: for decoding (stop strings, eos)
        G: group size — completions per prompt
        max_new_tokens: max completion length
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        profile: add NVTX annotations

    Returns:
        dict with:
            full_ids: (batch * G, full_seq_len) — prompt + completion
            full_mask: (batch * G, full_seq_len)
            prompt_lens: (batch * G,) — original prompt lengths
            completions: list[str] — decoded completions
            old_logprobs: (batch * G,) — sequence log-probs under current policy
    """
    if profile:
        torch.cuda.nvtx.range_push("rollout_generation")

    model.eval()
    batch_size = prompt_ids.shape[0]

    # Expand each prompt G times: [p1, p1, p1, p1, p2, p2, p2, p2, ...]
    expanded_ids = prompt_ids.repeat_interleave(G, dim=0)       # (B*G, prompt_len)
    expanded_mask = attention_mask.repeat_interleave(G, dim=0)   # (B*G, prompt_len)

    # Generate through FSDP forward (not model.module.generate which bypasses FSDP)
    output_ids = generate_with_fsdp(
        model=model,
        input_ids=expanded_ids,
        attention_mask=expanded_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Build full sequences and masks
    full_seq_len = output_ids.shape[1]
    full_mask = torch.ones_like(output_ids)

    # Mark padding in the prompt region
    prompt_len_per_seq = expanded_mask.sum(dim=1)  # actual prompt lengths (non-pad)
    for i in range(output_ids.shape[0]):
        pad_len = expanded_ids.shape[1] - prompt_len_per_seq[i].item()
        if pad_len > 0:
            full_mask[i, :pad_len] = 0

    # Decode completions and apply stop strings
    completions = []
    prompt_lens = prompt_len_per_seq.clone()

    for i in range(output_ids.shape[0]):
        plen = int(prompt_lens[i].item())
        # Account for any extra padding in prompt
        orig_prompt_len = expanded_ids.shape[1]
        comp_ids = output_ids[i, orig_prompt_len:]
        text = tokenizer.decode(comp_ids, skip_special_tokens=True)

        # Stop strings
        for stop in ["\n\nQuestion:", "\n\nProblem:", "\n\n\n"]:
            if stop in text:
                text = text[:text.index(stop)]

        completions.append(text)

    # Compute log-probs of the generated sequences under current policy
    # Process in chunks to avoid OOM (full batch logits = B*G × seq_len × vocab)
    if profile:
        torch.cuda.nvtx.range_push("rollout_logprobs")

    prompt_lens_tensor = torch.tensor(
        [expanded_ids.shape[1]] * output_ids.shape[0], device=output_ids.device
    )
    chunk_size = 2
    old_logprobs_list = []

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, output_ids.shape[0], chunk_size):
            chunk_ids = output_ids[i:i+chunk_size]
            chunk_mask = full_mask[i:i+chunk_size]
            chunk_prompt_lens = prompt_lens_tensor[i:i+chunk_size]

            chunk_lp = get_per_token_logprobs(model, chunk_ids, chunk_mask)
            chunk_seq_lp = sequence_logprobs(chunk_lp, chunk_mask, chunk_prompt_lens)
            old_logprobs_list.append(chunk_seq_lp)
            del chunk_lp

    old_logprobs = torch.cat(old_logprobs_list, dim=0)

    if profile:
        torch.cuda.nvtx.range_pop()  # rollout_logprobs

    model.train()

    if profile:
        torch.cuda.nvtx.range_pop()  # rollout_generation

    return {
        "full_ids": output_ids,
        "full_mask": full_mask,
        "prompt_lens": torch.tensor([expanded_ids.shape[1]] * output_ids.shape[0],
                                     device=output_ids.device),
        "completions": completions,
        "old_logprobs": old_logprobs.detach(),
    }


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_rollout_rewards(
    completions: list[str],
    raw_answers: list[str],
    G: int,
    dataset: str = "gsm8k",
    profile: bool = False,
) -> torch.Tensor:
    """Compute rewards for all rollouts.

    Args:
        completions: list of B*G decoded completions
        raw_answers: list of B ground truth answers (one per prompt)
        G: group size
        dataset: "gsm8k" or "math500"
    Returns:
        rewards: (B, G) tensor
    """
    if profile:
        torch.cuda.nvtx.range_push("reward_computation")

    batch_size = len(raw_answers)
    rewards = torch.zeros(batch_size, G)

    for i in range(batch_size):
        for g in range(G):
            idx = i * G + g
            completion = completions[idx]
            gt = raw_answers[i]

            rewards[i, g] = compute_reward(completion, gt, dataset=dataset)

    if profile:
        torch.cuda.nvtx.range_pop()

    return rewards


# ---------------------------------------------------------------------------
# RL training step
# ---------------------------------------------------------------------------

def rl_step(
    policy_model: FSDP,
    rollout_data: dict,
    rewards: torch.Tensor,
    config: dict,
    profile: bool = False,
) -> dict:
    """Single RL policy update step.

    Args:
        policy_model: trainable FSDP model
        rollout_data: from generate_rollouts()
        rewards: (B, G) tensor
        config: Dr.GRPO config section
        profile: NVTX annotations

    Returns:
        dict with loss, surrogate loss, and ratio statistics
    """
    device = next(policy_model.parameters()).device
    rewards = rewards.to(device)

    # ---- Compute advantages ----
    if profile:
        torch.cuda.nvtx.range_push("advantage_computation")

    advantages = compute_advantages_dr_grpo(rewards)

    if profile:
        torch.cuda.nvtx.range_pop()

    full_ids = rollout_data["full_ids"]
    full_mask = rollout_data["full_mask"]
    prompt_lens = rollout_data["prompt_lens"]
    old_logprobs = rollout_data["old_logprobs"]

    # Flatten advantages: (B, G) → (B*G,)
    advantages_flat = advantages.reshape(-1).to(device)

    # ---- Compute policy loss ----
    if profile:
        torch.cuda.nvtx.range_push("policy_loss_backward")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss_dict = dr_grpo_loss(
            policy_model=policy_model,
            input_ids=full_ids,
            attention_mask=full_mask,
            old_logprobs=old_logprobs,
            advantages=advantages_flat,
            prompt_lens=prompt_lens,
            clip_eps=config.get("clip_eps", 0.2),
        )

    loss_dict["loss"].backward()

    if profile:
        torch.cuda.nvtx.range_pop()

    return loss_dict


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

    # ---- Config parsing ----
    model_name = config["model"]["name"]
    sft_checkpoint = config["model"]["sft_checkpoint"]
    num_steps = config["training"]["num_steps"]
    warmup_steps = config["training"]["warmup_steps"]

    max_lr = config["optimizer"]["lr"]
    min_lr = config["scheduler"]["min_lr"]

    algo_config = config["dr_grpo"]
    G = algo_config["group_size"]
    max_rollout_len = algo_config["max_rollout_len"]
    num_prompts = algo_config["num_prompts_per_step"]
    temperature = algo_config["temperature"]
    top_p = algo_config["top_p"]

    log_every = config["logging"]["log_every"]
    save_every = config["logging"]["save_every"]
    output_dir = config["logging"]["output_dir"]

    if rank == 0:
        print(f"\n{'='*60}")
        print("RL Training — DR_GRPO")
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  SFT checkpoint: {sft_checkpoint}")
        print(f"  GPUs: {world_size}")
        print(f"  G={G} | prompts/step={num_prompts} | "
              f"rollouts/step={num_prompts * G}")
        print(f"  max_rollout_len={max_rollout_len} | temp={temperature}")
        print(f"  Steps: {num_steps} | lr: {max_lr}")
        print(f"  Profile: {profile_mode}")
        print(f"{'='*60}")

        wandb.init(
            project="reasoning-rl",
            name=f"dr-grpo-{model_name.split('/')[-1]}",
            config=config,
        )

    # ---- Load tokenizer ----
    tokenizer = load_tokenizer(model_name)

    # ---- Load policy model (from SFT checkpoint) ----
    policy = load_model(sft_checkpoint, dtype=torch.bfloat16)
    policy = wrap_model_fsdp(
        policy,
        mixed_precision=config["fsdp"]["mixed_precision"],
        activation_checkpointing=config["fsdp"]["activation_checkpointing"],
        forward_prefetch=config["fsdp"]["forward_prefetch"],
    )

    if rank == 0:
        log_memory("after policy load")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=max_lr,
        weight_decay=config["optimizer"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )

    # ---- Dataloader ----
    dataloader = create_rl_dataloader(
        tokenizer=tokenizer,
        dataset_name=config["data"]["dataset"],
        split=config["data"]["split"],
        batch_size=num_prompts,
        num_workers=4,
        distributed=True,
    )

    # ---- Profiling ----
    profiler = ProfilerControl(
        warmup_steps=config["profiling"].get("warmup_steps", 10),
        capture_steps=config["profiling"].get("capture_steps", 20),
        enabled=profile_mode,
    )
    tracker = MetricTracker(log_dir=output_dir)

    if rank == 0:
        log_memory("before training")
        reset_peak_memory()

    # ---- Training loop ----
    data_iter = iter(dataloader)
    policy.train()

    pbar = tqdm(range(num_steps), desc="DR_GRPO", disable=(rank != 0))
    for step in pbar:
        profiler.step(step)

        if profile_mode:
            torch.cuda.nvtx.range_push(f"dr_grpo_step_{step}")

        # LR schedule
        lr = get_cosine_lr(step, num_steps, warmup_steps, max_lr, min_lr)
        set_lr(optimizer, lr)

        # Get batch of prompts
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt_ids = batch["prompt_ids"].cuda()
        prompt_mask = batch["attention_mask"].cuda()
        raw_answers = batch["raw_answers"]

        # ---- Generate rollouts ----
        rollout_data = generate_rollouts(
            model=policy,
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            tokenizer=tokenizer,
            G=G,
            max_new_tokens=max_rollout_len,
            temperature=temperature,
            top_p=top_p,
            profile=profile_mode,
        )

        # ---- Compute rewards ----
        rewards = compute_rollout_rewards(
            completions=rollout_data["completions"],
            raw_answers=raw_answers,
            G=G,
            dataset="gsm8k",
            profile=profile_mode,
        )

        # ---- Policy update ----
        loss_dict = rl_step(
            policy_model=policy,
            rollout_data=rollout_data,
            rewards=rewards,
            config=algo_config,
            profile=profile_mode,
        )

        if profile_mode:
            torch.cuda.nvtx.range_push("optimizer_step")

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if profile_mode:
            torch.cuda.nvtx.range_pop()  # optimizer_step

        optimizer.zero_grad(set_to_none=True)

        if profile_mode:
            torch.cuda.nvtx.range_pop()  # dr_grpo_step

        # ---- Logging ----
        if rank == 0:
            reward_mean = rewards.mean().item()
            reward_std = rewards.std().item()

            tracker.update(
                step=step,
                loss=loss_dict["loss"].item(),
                surrogate_loss=loss_dict["surrogate_loss"].item(),
                ratio_mean=loss_dict["ratio_mean"].item(),
                reward_mean=reward_mean,
                reward_std=reward_std,
                lr=lr,
                grad_norm=grad_norm.item(),
            )

            pbar.set_postfix(
                loss=f"{loss_dict['loss'].item():.3f}",
                reward=f"{reward_mean:.2f}",
                mem=f"{torch.cuda.max_memory_allocated() / 1e9:.1f}G",
            )

            wandb.log({
                "loss": loss_dict["loss"].item(),
                "surrogate_loss": loss_dict["surrogate_loss"].item(),
                "ratio_mean": loss_dict["ratio_mean"].item(),
                "reward/mean": reward_mean,
                "reward/std": reward_std,
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "memory/peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            }, step=step)

            if step % log_every == 0 or step == num_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1e9
                tracker.update(step=step, peak_mem_gb=mem)
                tracker.log(step, prefix="DR_GRPO")

        # ---- Checkpointing ----
        if (step + 1) % save_every == 0:
            if rank == 0:
                print(f"\n[Step {step}] Saving checkpoint...")
            save_hf_checkpoint(
                model=policy, tokenizer=tokenizer,
                step=step + 1, output_dir=output_dir,
                model_name=model_name, rank=rank,
            )

        dist.barrier()

    # ---- Final checkpoint ----
    save_hf_checkpoint(
        model=policy, tokenizer=tokenizer,
        step=num_steps, output_dir=output_dir + "/final",
        model_name=model_name, rank=rank,
    )

    profiler.stop()

    gpu_memory = gather_all_gpu_memory()
    if rank == 0:
        tracker.save(filename="dr_grpo_metrics.json")
        log_memory("end of training")
        save_memory_report(
            "results/memory_dr_grpo.json",
            stage="dr_grpo",
            gpus=gpu_memory,
            extra={
                "model": model_name,
                "sft_checkpoint": sft_checkpoint,
                "group_size": G,
                "max_rollout_len": max_rollout_len,
                "num_prompts_per_step": num_prompts,
                "num_gpus": dist.get_world_size(),
            },
        )

        print(f"\n{'='*60}")
        print("DR_GRPO TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Steps: {num_steps}")
        print(f"  Final reward mean: {reward_mean:.3f}")
        print(f"  Checkpoint: {output_dir}/final/step_{num_steps}")
        print(f"  Metrics: {output_dir}/dr_grpo_metrics.json")
        print(f"{'='*60}")

        wandb.finish()

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--profile", action="store_true")

    # CLI overrides
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.num_steps is not None:
        config["training"]["num_steps"] = args.num_steps
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    if args.output_dir is not None:
        config["logging"]["output_dir"] = args.output_dir

    train(config, profile_mode=args.profile)


if __name__ == "__main__":
    main()
