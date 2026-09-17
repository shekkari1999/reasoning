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


def sequence_logprobs(
    per_token_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Sum log probabilities over completion tokens only."""
    shift_mask = attention_mask[:, 1:]
    positions = torch.arange(
        shift_mask.shape[1], device=shift_mask.device
    ).unsqueeze(0)
    completion_mask = (
        positions >= (prompt_lens.unsqueeze(1) - 1)
    ) & shift_mask.bool()
    return (per_token_logprobs * completion_mask).sum(dim=-1)


def compute_advantages_dr_grpo(rewards: torch.Tensor) -> torch.Tensor:
    """Center rewards within each prompt group without std normalization."""
    return rewards - rewards.mean(dim=-1, keepdim=True)


def dr_grpo_loss(
    policy_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    prompt_lens: torch.Tensor,
    clip_eps: float = 0.2,
) -> dict:
    """Compute the clipped Dr.GRPO surrogate loss without a KL term."""
    new_per_token = get_per_token_logprobs(
        policy_model, input_ids, attention_mask
    )
    new_logprobs = sequence_logprobs(
        new_per_token, attention_mask, prompt_lens
    )

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = -torch.min(
        ratio * advantages, clipped_ratio * advantages
    ).mean()

    return {
        "loss": surrogate,
        "surrogate_loss": surrogate.detach(),
        "ratio_mean": ratio.mean().detach(),
    }
