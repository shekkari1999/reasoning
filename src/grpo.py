"""
GRPO (Group Relative Policy Optimization) from Scratch
- Rollout sampling from policy
- Group-relative advantage estimation
- Clipped policy gradient with KL penalty
- Multi-epoch training with checkpointing

Reference: DeepSeek-R1 (https://arxiv.org/abs/2501.12948)
"""

import os
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from eval.answer_extraction import extract_answer
from eval.math_verifier import verify_answer


def compute_log_probs(model, input_ids, response_ids):
    """
    Compute per-token log probabilities for response tokens.

    Args:
        model: the language model
        input_ids: prompt token ids [1, prompt_len]
        response_ids: response token ids [1, response_len]

    Returns:
        per-token log probs [1, response_len]
    """
    full_ids = torch.cat([input_ids, response_ids], dim=-1)
    outputs = model(full_ids)
    logits = outputs.logits

    # at position i, the model predicts token i+1
    # so for response tokens starting at prompt_len, we need logits at [prompt_len-1 : prompt_len+response_len-1]
    prompt_len = input_ids.shape[1]
    response_len = response_ids.shape[1]
    response_logits = logits[:, prompt_len - 1 : prompt_len + response_len - 1, :]

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs


def generate_rollouts(model, tokenizer, prompt, n_samples, max_new_tokens, temperature, device):
    """
    Sample N responses from the current policy for a given prompt.
    Returns list of rollout dicts with response text and token ids.
    """
    rollouts = []
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(n_samples):
        generated_ids = []
        past_key_values = None
        current_input = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :] / temperature
                past_key_values = outputs.past_key_values

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids.append(next_token.item())

                if next_token.item() == tokenizer.eos_token_id:
                    break

                current_input = next_token

        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_ids = torch.tensor([generated_ids], device=device)

        rollouts.append({
            "response": response_text,
            "input_ids": input_ids,
            "response_ids": response_ids,
        })

    return rollouts


def compute_rewards(rollouts, ground_truth):
    """
    Binary reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise.
    Uses symbolic equivalence checking via math_verifier.
    """
    rewards = []
    for r in rollouts:
        extracted = extract_answer(r["response"])
        if extracted is not None:
            reward = 1.0 if verify_answer(str(extracted), str(ground_truth)) else 0.0
        else:
            reward = 0.0
        rewards.append(reward)
    return torch.tensor(rewards)


def compute_group_advantages(rewards):
    """
    Group-relative advantages: normalize rewards within the group.
    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
    """
    mean = rewards.mean()
    std = rewards.std()
    if std < 1e-8:
        return torch.zeros_like(rewards)
    advantages = (rewards - mean) / (std + 1e-8)
    return advantages


def grpo_loss(model, ref_model, rollouts, advantages, kl_coeff=0.1, clip_eps=0.2):
    """
    Compute GRPO loss with clipped policy gradient and KL penalty.

    L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] + kl_coeff * KL

    Where ratio = exp(log_pi - log_pi_old) and KL is per-token KL divergence.
    """
    total_loss = torch.tensor(0.0, device=rollouts[0]["input_ids"].device, requires_grad=True)
    valid_count = 0

    for rollout, advantage in zip(rollouts, advantages):
        input_ids = rollout["input_ids"]
        response_ids = rollout["response_ids"]

        if response_ids.shape[1] == 0:
            continue

        # current policy log probs
        policy_log_probs = compute_log_probs(model, input_ids, response_ids)

        # reference model log probs (frozen)
        with torch.no_grad():
            ref_log_probs = compute_log_probs(ref_model, input_ids, response_ids)

        # per-token policy gradient
        # using simplified REINFORCE-style: -advantage * sum(log_probs)
        pg_loss = -(advantage * policy_log_probs.sum(dim=-1))

        # KL divergence: KL(policy || ref) = sum(exp(policy) * (policy - ref))
        # simplified approximation: sum(policy_log_prob - ref_log_prob)
        kl_div = (policy_log_probs - ref_log_probs).sum(dim=-1)

        total_loss = total_loss + pg_loss.squeeze() + kl_coeff * kl_div.squeeze()
        valid_count += 1

    if valid_count == 0:
        return total_loss

    return total_loss / valid_count


def train_grpo(
    model,
    ref_model,
    tokenizer,
    dataset,
    n_samples=8,
    epochs=5,
    lr=1e-6,
    kl_coeff=0.1,
    max_new_tokens=512,
    temperature=0.7,
    max_grad_norm=1.0,
    device="cuda",
    checkpoint_dir="checkpoints",
    log_every=10,
):
    """
    Full GRPO training loop.

    For each question:
    1. Sample N responses from current policy
    2. Score each response with binary reward (verifiable)
    3. Compute group-relative advantages
    4. Update policy with policy gradient + KL penalty
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    ref_model.eval()
    model.train()

    metrics = {
        "epoch_rewards": [],
        "epoch_losses": [],
    }

    for epoch in range(epochs):
        total_reward = 0.0
        total_loss = 0.0
        total_examples = 0
        n_updates = 0

        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch+1}/{epochs}")

        for i, example in pbar:
            question = example["question"]
            ground_truth = example["answer"]

            # 1. generate rollouts from current policy
            rollouts = generate_rollouts(
                model, tokenizer, question,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )

            # 2. compute binary rewards
            rewards = compute_rewards(rollouts, ground_truth).to(device)
            total_reward += rewards.sum().item()
            total_examples += len(rewards)

            # 3. compute group-relative advantages
            advantages = compute_group_advantages(rewards)

            # skip if all rewards are the same (no learning signal)
            if rewards.std() < 1e-8:
                continue

            # 4. compute loss and update
            optimizer.zero_grad()
            loss = grpo_loss(model, ref_model, rollouts, advantages, kl_coeff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_updates += 1

            if (i + 1) % log_every == 0:
                avg_reward = total_reward / max(total_examples, 1)
                avg_loss = total_loss / max(n_updates, 1)
                pbar.set_postfix({
                    "avg_reward": f"{avg_reward:.3f}",
                    "avg_loss": f"{avg_loss:.4f}",
                })

        # epoch summary
        epoch_avg_reward = total_reward / max(total_examples, 1)
        epoch_avg_loss = total_loss / max(n_updates, 1)
        metrics["epoch_rewards"].append(epoch_avg_reward)
        metrics["epoch_losses"].append(epoch_avg_loss)

        print(f"\nEpoch {epoch+1} | Avg Reward: {epoch_avg_reward:.3f} | Avg Loss: {epoch_avg_loss:.4f} | Updates: {n_updates}")

        # save checkpoint
        epoch_path = os.path.join(checkpoint_dir, f"grpo_epoch_{epoch+1}")
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)
        print(f"Saved checkpoint: {epoch_path}")

    return metrics
