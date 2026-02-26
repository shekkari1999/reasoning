"""
RL loss functions for GRPO, Dr. GRPO, and DAPO.

All three share the same structure:
  1. Compute log-probs of rollout tokens under current policy
  2. Compute advantages from rewards
  3. Compute clipped surrogate loss
  4. (Optional) Add KL penalty

They differ in:
  - GRPO:    symmetric clipping + KL penalty + normalized advantages
  - Dr.GRPO: symmetric clipping + NO KL + mean-only advantages
  - DAPO:    asymmetric clipping + NO KL + dynamic sampling + overlong penalty
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_per_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities under a model.
    
    Args:
        model: FSDP-wrapped language model
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
    
    Returns:
        log_probs: (batch, seq_len - 1) — log p(token_t | tokens_<t)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    # Shift: predict token t from position t-1
    shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    shift_labels = input_ids[:, 1:]   # (batch, seq_len-1)

    # Log-softmax over vocab, then gather the log-prob of the actual token
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out padding tokens
    shift_mask = attention_mask[:, 1:]  # (batch, seq_len-1)
    token_log_probs = token_log_probs * shift_mask

    return token_log_probs


def sequence_logprobs(
    per_token_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Sum per-token log-probs over the COMPLETION only (not the prompt).
    
    Args:
        per_token_logprobs: (batch, seq_len - 1)
        attention_mask: (batch, seq_len)
        prompt_lens: (batch,) — number of prompt tokens per sequence
    
    Returns:
        seq_logprobs: (batch,) — sum of log-probs over completion tokens
    """
    shift_mask = attention_mask[:, 1:]  # align with shifted log-probs
    batch_size, seq_len = shift_mask.shape

    # Create mask that zeros out prompt tokens
    positions = torch.arange(seq_len, device=shift_mask.device).unsqueeze(0)
    # prompt_lens - 1 because of the shift
    completion_mask = (positions >= (prompt_lens.unsqueeze(1) - 1)) & (shift_mask.bool())
    completion_mask = completion_mask.float()

    return (per_token_logprobs * completion_mask).sum(dim=-1)


# ---------------------------------------------------------------------------
# Advantage computation variants
# ---------------------------------------------------------------------------

def compute_advantages_grpo(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """GRPO advantage: group-normalized.
    
    For each group of G completions for the same prompt:
        A_i = (r_i - mean(r)) / (std(r) + eps)
    
    Args:
        rewards: (num_prompts, G) — rewards for each completion
    
    Returns:
        advantages: (num_prompts, G) — normalized advantages
    """
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages


def compute_advantages_dr_grpo(
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Dr. GRPO advantage: mean-only, no std normalization.
    
    A_i = r_i - mean(r)
    
    Simpler, avoids potential issues with std normalization
    when rewards have low variance.
    """
    mean_r = rewards.mean(dim=-1, keepdim=True)
    return rewards - mean_r


def compute_advantages_dapo(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """DAPO advantage: same as GRPO (group-normalized).
    
    The DAPO paper uses the same normalization as GRPO.
    The key DAPO difference is in dynamic sampling (handled outside the loss).
    """
    return compute_advantages_grpo(rewards, eps=eps)


# ---------------------------------------------------------------------------
# GRPO Loss
# ---------------------------------------------------------------------------

def grpo_loss(
    policy_model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    prompt_lens: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.1,
) -> dict:
    """Vanilla GRPO loss with symmetric clipping and KL penalty.
    
    L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] + beta * KL(policy || ref)
    
    Args:
        policy_model: Current policy (FSDP-wrapped, trainable)
        ref_model: Frozen reference model (FSDP-wrapped, no grad)
        input_ids: (batch, seq_len) — full sequences (prompt + completion)
        attention_mask: (batch, seq_len)
        old_logprobs: (batch,) — sequence log-probs from rollout generation
        advantages: (batch,) — pre-computed advantages (flattened from (num_prompts, G))
        prompt_lens: (batch,) — prompt lengths for masking
        clip_eps: Symmetric clipping epsilon
        kl_beta: KL penalty coefficient
    
    Returns:
        Dict with loss, surrogate_loss, kl, ratio_mean
    """
    # Current policy log-probs
    new_per_token = get_per_token_logprobs(policy_model, input_ids, attention_mask)
    new_logprobs = sequence_logprobs(new_per_token, attention_mask, prompt_lens)

    # Reference log-probs (no grad)
    with torch.no_grad():
        ref_per_token = get_per_token_logprobs(ref_model, input_ids, attention_mask)
        ref_logprobs = sequence_logprobs(ref_per_token, attention_mask, prompt_lens)

    # Policy ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # Clipped surrogate
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # KL divergence: KL(policy || ref) ≈ E[log(policy) - log(ref)]
    kl = (new_logprobs - ref_logprobs).mean()

    # Total loss
    loss = surrogate + kl_beta * kl

    return {
        "loss": loss,
        "surrogate_loss": surrogate.detach(),
        "kl": kl.detach(),
        "ratio_mean": ratio.mean().detach(),
    }


# ---------------------------------------------------------------------------
# Dr. GRPO Loss
# ---------------------------------------------------------------------------

def dr_grpo_loss(
    policy_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    prompt_lens: torch.Tensor,
    clip_eps: float = 0.2,
) -> dict:
    """Dr. GRPO loss: NO KL penalty, NO reference model.
    
    L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
    
    Relies solely on clipping to prevent policy collapse.
    Saves ~3GB/GPU by not loading a reference model.
    """
    # Current policy log-probs
    new_per_token = get_per_token_logprobs(policy_model, input_ids, attention_mask)
    new_logprobs = sequence_logprobs(new_per_token, attention_mask, prompt_lens)

    # Policy ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # Clipped surrogate (same as GRPO, just no KL)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    loss = surrogate

    return {
        "loss": loss,
        "surrogate_loss": surrogate.detach(),
        "kl": torch.tensor(0.0),  # no KL computed
        "ratio_mean": ratio.mean().detach(),
    }


# ---------------------------------------------------------------------------
# DAPO Loss
# ---------------------------------------------------------------------------

def dapo_loss(
    policy_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    prompt_lens: torch.Tensor,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
) -> dict:
    """DAPO loss: asymmetric clipping, no KL, no reference model.
    
    Key insight: use a LOOSER clip on the upside (positive advantages)
    to allow the policy to explore more aggressively when it finds
    good solutions. Tighter clip on the downside to prevent catastrophic
    forgetting.
    
    L = -E[min(ratio * A, clip_asym(ratio) * A)]
    
    where clip_asym uses eps_low for A < 0 and eps_high for A >= 0
    """
    # Current policy log-probs
    new_per_token = get_per_token_logprobs(policy_model, input_ids, attention_mask)
    new_logprobs = sequence_logprobs(new_per_token, attention_mask, prompt_lens)

    # Policy ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # Asymmetric clipping
    # For positive advantages: clip to [1 - eps_low, 1 + eps_high] (looser upside)
    # For negative advantages: clip to [1 - eps_low, 1 + eps_high] (tighter downside)
    clip_low = 1.0 - clip_eps_low
    clip_high = 1.0 + clip_eps_high

    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

    surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    loss = surrogate

    return {
        "loss": loss,
        "surrogate_loss": surrogate.detach(),
        "kl": torch.tensor(0.0),
        "ratio_mean": ratio.mean().detach(),
    }


def filter_dynamic_sampling(
    rewards: torch.Tensor,
) -> torch.Tensor:
    """DAPO dynamic sampling: create mask for prompts with non-zero reward variance.
    
    If all G completions for a prompt got the same reward (all correct or all wrong),
    the advantages are all zero → zero gradient signal → wasted compute.
    Filter these out.
    
    Args:
        rewards: (num_prompts, G)
    
    Returns:
        mask: (num_prompts,) — True for prompts to keep
    """
    reward_std = rewards.std(dim=-1)
    return reward_std > 1e-8  # keep prompts with non-zero variance
