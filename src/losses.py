"""Loss and advantage utilities for the Dr.GRPO training loop."""

import torch
import torch.nn.functional as F


def get_per_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return log p(token_t | tokens_<t) for each non-padding token."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_log_probs * shift_mask


def build_completion_mask(
    attention_mask: torch.Tensor,
    prompt_width: int,
) -> torch.Tensor:
    """Mask shifted-token positions that belong to generated completions."""
    shift_mask = attention_mask[:, 1:].bool()
    positions = torch.arange(shift_mask.shape[1], device=shift_mask.device)
    return shift_mask & (positions.unsqueeze(0) >= prompt_width - 1)


def compute_advantages_dr_grpo(rewards: torch.Tensor) -> torch.Tensor:
    """Center rewards within each prompt group without std normalization."""
    return rewards - rewards.mean(dim=-1, keepdim=True)


def dr_grpo_loss(
    policy_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    old_per_token_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    max_completion_length: int,
    clip_eps: float = 0.2,
) -> dict:
    """Compute token-level clipped Dr.GRPO with a fixed-length denominator."""
    new_per_token = get_per_token_logprobs(
        policy_model, input_ids, attention_mask
    )
    ratio = torch.exp(new_per_token - old_per_token_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    token_advantages = advantages.unsqueeze(1)
    per_token_loss = -torch.min(
        ratio * token_advantages,
        clipped_ratio * token_advantages,
    )
    mask = completion_mask.to(per_token_loss.dtype)
    denominator = input_ids.shape[0] * max_completion_length
    surrogate = (per_token_loss * mask).sum() / denominator
    valid_tokens = mask.sum().clamp_min(1.0)
    clipped = (torch.abs(ratio - 1.0) > clip_eps).to(mask.dtype)

    return {
        "loss": surrogate,
        "surrogate_loss": surrogate.detach(),
        "ratio_mean": ((ratio * mask).sum() / valid_tokens).detach(),
        "clipped_fraction": ((clipped * mask).sum() / valid_tokens).detach(),
    }
