"""
Reinforcement Learning Training — GRPO / Dr.GRPO / DAPO

Unified training loop. The three algorithms share rollout generation,
reward computation, and FSDP infrastructure. They differ only in:
  - Loss function (from losses.py)
  - Whether a reference model is loaded
  - Advantage computation
  - (DAPO) Dynamic sampling and overlong penalty

Usage:
    # GRPO (with KL penalty + reference model)
    torchrun --nproc_per_node=2 src/rl_train.py --algo grpo --config configs/grpo_config.yaml

    # Dr. GRPO (no KL, no ref model, simplified advantages)
    torchrun --nproc_per_node=2 src/rl_train.py --algo dr_grpo --config configs/dr_grpo_config.yaml

    # DAPO (asymmetric clipping, dynamic sampling, overlong penalty)
    torchrun --nproc_per_node=2 src/rl_train.py --algo dapo --config configs/dapo_config.yaml

    # Profile any variant
    nsys profile --trace=cuda,nvtx,nccl --capture-range=cudaProfilerApi \
        --output=profiles/grpo \
        torchrun --nproc_per_node=2 src/rl_train.py --algo grpo --config configs/grpo_config.yaml --profile
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
from src.model import (
    load_model, load_tokenizer, wrap_model_fsdp,
    load_reference_model, save_hf_checkpoint,
)
from src.data import create_rl_dataloader
from src.losses import (
    get_per_token_logprobs,
    sequence_logprobs,
    compute_advantages_grpo,
    compute_advantages_dr_grpo,
    compute_advantages_dapo,
    grpo_loss,
    dr_grpo_loss,
    dapo_loss,
    filter_dynamic_sampling,
)
from src.rewards import (
    compute_reward,
    compute_reward_with_overlong_penalty,
)
from src.profiling_utils import (
    nvtx_range,
    ProfilerControl,
    MetricTracker,
    log_memory,
    reset_peak_memory,
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
            completion_lens: (batch * G,) — completion lengths in tokens
            old_logprobs: (batch * G,) — sequence log-probs under current policy
    """
    if profile:
        torch.cuda.nvtx.range_push("rollout_generation")

    model.eval()
    batch_size = prompt_ids.shape[0]

    # Expand each prompt G times: [p1, p1, p1, p1, p2, p2, p2, p2, ...]
    expanded_ids = prompt_ids.repeat_interleave(G, dim=0)       # (B*G, prompt_len)
    expanded_mask = attention_mask.repeat_interleave(G, dim=0)   # (B*G, prompt_len)

    # Generate
    output_ids = model.module.generate(
        input_ids=expanded_ids,
        attention_mask=expanded_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
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
    completion_lens = []
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
        comp_tokens = tokenizer.encode(text, add_special_tokens=False)
        completion_lens.append(len(comp_tokens))

    completion_lens = torch.tensor(completion_lens, device=output_ids.device)

    # Compute log-probs of the generated sequences under current policy
    if profile:
        torch.cuda.nvtx.range_push("rollout_logprobs")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        per_token_lp = get_per_token_logprobs(model, output_ids, full_mask)
        old_logprobs = sequence_logprobs(
            per_token_lp, full_mask,
            torch.tensor([expanded_ids.shape[1]] * output_ids.shape[0],
                         device=output_ids.device)
        )

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
        "completion_lens": completion_lens,
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
    algo: str = "grpo",
    completion_lens: torch.Tensor = None,
    max_rollout_len: int = 512,
    profile: bool = False,
) -> torch.Tensor:
    """Compute rewards for all rollouts.

    Args:
        completions: list of B*G decoded completions
        raw_answers: list of B ground truth answers (one per prompt)
        G: group size
        dataset: "gsm8k" or "math500"
        algo: "grpo", "dr_grpo", or "dapo"
        completion_lens: token lengths for overlong penalty (DAPO only)
        max_rollout_len: max length for overlong check (DAPO only)

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

            if algo == "dapo" and completion_lens is not None:
                reward = compute_reward_with_overlong_penalty(
                    completion, gt,
                    completion_len=completion_lens[idx].item(),
                    max_len=max_rollout_len,
                    dataset=dataset,
                )
            else:
                reward = compute_reward(completion, gt, dataset=dataset)

            rewards[i, g] = reward

    if profile:
        torch.cuda.nvtx.range_pop()

    return rewards


# ---------------------------------------------------------------------------
# RL training step
# ---------------------------------------------------------------------------

def rl_step(
    algo: str,
    policy_model: FSDP,
    ref_model: FSDP,
    rollout_data: dict,
    rewards: torch.Tensor,
    G: int,
    config: dict,
    profile: bool = False,
) -> dict:
    """Single RL policy update step.

    Args:
        algo: "grpo", "dr_grpo", or "dapo"
        policy_model: trainable FSDP model
        ref_model: frozen reference (None for dr_grpo/dapo)
        rollout_data: from generate_rollouts()
        rewards: (B, G) tensor
        G: group size
        config: algorithm-specific config section
        profile: NVTX annotations

    Returns:
        dict with loss, surrogate, kl, reward stats
    """
    device = next(policy_model.parameters()).device
    rewards = rewards.to(device)

    # ---- Compute advantages ----
    if profile:
        torch.cuda.nvtx.range_push("advantage_computation")

    if algo == "grpo":
        advantages = compute_advantages_grpo(rewards)
    elif algo == "dr_grpo":
        advantages = compute_advantages_dr_grpo(rewards)
    elif algo == "dapo":
        advantages = compute_advantages_dapo(rewards)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    if profile:
        torch.cuda.nvtx.range_pop()

    # ---- DAPO dynamic sampling: filter zero-variance groups ----
    if algo == "dapo" and config.get("dynamic_sampling", False):
        keep_mask = filter_dynamic_sampling(rewards)
        num_skipped = (~keep_mask).sum().item()

        if keep_mask.sum() == 0:
            # All groups have zero variance — skip this step
            return {
                "loss": torch.tensor(0.0),
                "surrogate_loss": torch.tensor(0.0),
                "kl": torch.tensor(0.0),
                "ratio_mean": torch.tensor(1.0),
                "prompts_skipped": num_skipped,
                "skipped_step": True,
            }

        # Filter to kept groups only
        advantages = advantages[keep_mask]
        rewards_kept = rewards[keep_mask]

        # Filter rollout data — need to select the right indices
        kept_indices = []
        for i, keep in enumerate(keep_mask):
            if keep:
                for g in range(G):
                    kept_indices.append(i * G + g)
        kept_indices = torch.tensor(kept_indices, device=device)

        full_ids = rollout_data["full_ids"][kept_indices]
        full_mask = rollout_data["full_mask"][kept_indices]
        prompt_lens = rollout_data["prompt_lens"][kept_indices]
        old_logprobs = rollout_data["old_logprobs"][kept_indices]
    else:
        num_skipped = 0
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
        if algo == "grpo":
            loss_dict = grpo_loss(
                policy_model=policy_model,
                ref_model=ref_model,
                input_ids=full_ids,
                attention_mask=full_mask,
                old_logprobs=old_logprobs,
                advantages=advantages_flat,
                prompt_lens=prompt_lens,
                clip_eps=config.get("clip_eps", 0.2),
                kl_beta=config.get("kl_beta", 0.1),
            )
        elif algo == "dr_grpo":
            loss_dict = dr_grpo_loss(
                policy_model=policy_model,
                input_ids=full_ids,
                attention_mask=full_mask,
                old_logprobs=old_logprobs,
                advantages=advantages_flat,
                prompt_lens=prompt_lens,
                clip_eps=config.get("clip_eps", 0.2),
            )
        elif algo == "dapo":
            loss_dict = dapo_loss(
                policy_model=policy_model,
                input_ids=full_ids,
                attention_mask=full_mask,
                old_logprobs=old_logprobs,
                advantages=advantages_flat,
                prompt_lens=prompt_lens,
                clip_eps_low=config.get("clip_eps_low", 0.2),
                clip_eps_high=config.get("clip_eps_high", 0.28),
            )

    loss_dict["loss"].backward()

    if profile:
        torch.cuda.nvtx.range_pop()

    loss_dict["prompts_skipped"] = num_skipped
    loss_dict["skipped_step"] = False
    return loss_dict


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, algo: str, profile_mode: bool = False):
    # ---- Distributed setup ----
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- Config parsing ----
    model_name = config["model"]["name"]
    sft_checkpoint = config["model"]["sft_checkpoint"]
    micro_batch_size = config["training"]["micro_batch_size"]
    accum_steps = config["training"]["gradient_accumulation_steps"]
    num_steps = config["training"]["num_steps"]
    warmup_steps = config["training"]["warmup_steps"]

    max_lr = config["optimizer"]["lr"]
    min_lr = config["scheduler"]["min_lr"]

    # Algorithm-specific config
    algo_config = config.get(algo, config.get("grpo", {}))
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
        print(f"RL Training — {algo.upper()}")
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  SFT checkpoint: {sft_checkpoint}")
        print(f"  GPUs: {world_size}")
        print(f"  G={G} | prompts/step={num_prompts} | "
              f"rollouts/step={num_prompts * G}")
        print(f"  max_rollout_len={max_rollout_len} | temp={temperature}")
        print(f"  Reference model: {use_ref_model}")
        print(f"  Steps: {num_steps} | lr: {max_lr}")
        print(f"  Profile: {profile_mode}")
        print(f"{'='*60}")

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

    # ---- Load reference model (GRPO only) ----
    ref_model = None
    if use_ref_model:
        ref_model = load_reference_model(
            sft_checkpoint,
            dtype=torch.bfloat16,
            mixed_precision=config["fsdp"]["mixed_precision"],
        )
        if rank == 0:
            log_memory("after ref model load")

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

    for step in range(num_steps):
        profiler.step(step)

        if profile_mode:
            torch.cuda.nvtx.range_push(f"{algo}_step_{step}")

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
            algo=algo,
            completion_lens=rollout_data["completion_lens"],
            max_rollout_len=max_rollout_len,
            profile=profile_mode,
        )

        # ---- Policy update ----
        loss_dict = rl_step(
            algo=algo,
            policy_model=policy,
            ref_model=ref_model,
            rollout_data=rollout_data,
            rewards=rewards,
            G=G,
            config=algo_config,
            profile=profile_mode,
        )

        if not loss_dict.get("skipped_step", False):
            # Gradient clipping
            if profile_mode:
                torch.cuda.nvtx.range_push("optimizer_step")

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            if profile_mode:
                torch.cuda.nvtx.range_pop()  # optimizer_step

        optimizer.zero_grad(set_to_none=True)

        if profile_mode:
            torch.cuda.nvtx.range_pop()  # algo_step

        # ---- Logging ----
        if rank == 0:
            reward_mean = rewards.mean().item()
            reward_std = rewards.std().item()

            tracker.update(
                step=step,
                loss=loss_dict["loss"].item(),
                surrogate_loss=loss_dict["surrogate_loss"].item(),
                kl=loss_dict["kl"].item(),
                ratio_mean=loss_dict["ratio_mean"].item(),
                reward_mean=reward_mean,
                reward_std=reward_std,
                lr=lr,
                grad_norm=grad_norm.item() if not loss_dict.get("skipped_step") and isinstance(grad_norm, torch.Tensor) else 0.0,
                prompts_skipped=loss_dict.get("prompts_skipped", 0),
            )

            if step % log_every == 0 or step == num_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1e9
                tracker.update(step=step, peak_mem_gb=mem)
                tracker.log(step, prefix=algo.upper())

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

    if rank == 0:
        tracker.save(filename=f"{algo}_metrics.json")
        log_memory("end of training")

        print(f"\n{'='*60}")
        print(f"{algo.upper()} TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Steps: {num_steps}")
        print(f"  Final reward mean: {reward_mean:.3f}")
        print(f"  Checkpoint: {output_dir}/final")
        print(f"  Metrics: {output_dir}/{algo}_metrics.json")
        print(f"{'='*60}")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Thin wrappers for each algorithm (can also be called directly)
# ---------------------------------------------------------------------------

def train_grpo(config, profile=False):
    train(config, algo="grpo", profile_mode=profile)

def train_dr_grpo(config, profile=False):
    train(config, algo="dr_grpo", profile_mode=profile)

def train_dapo(config, profile=False):
    train(config, algo="dapo", profile_mode=profile)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True,
                        choices=["grpo", "dr_grpo", "dapo"])
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

    train(config, algo=args.algo, profile_mode=args.profile)


if __name__ == "__main__":
    main()
