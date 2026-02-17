"""
Inference Engine
- Custom text generation with KV-cache
- Temperature and top-p (nucleus) sampling
- torch.compile optimization for faster generation
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def top_p_filter(logits, top_p):
    """Apply top-p (nucleus) filtering to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # remove tokens with cumulative prob above threshold (keep at least one token)
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_mask] = float("-inf")

    # scatter back to original ordering
    logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
    return logits


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=1.0,
    top_p=1.0,
    device="cuda",
    return_log_probs=False,
):
    """
    Generate text with KV-cache, temperature scaling, and top-p sampling.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        prompt: input text string
        max_new_tokens: max tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_p: nucleus sampling threshold
        device: cuda or cpu
        return_log_probs: if True, also return per-token log probabilities

    Returns:
        generated text (str), or (text, log_probs) if return_log_probs=True
    """
    if isinstance(prompt, str):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        input_ids = prompt.to(device)

    past_key_values = None
    generated_ids = []
    token_log_probs = []
    current_input = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # temperature scaling
            if temperature > 0:
                logits = logits / temperature
            else:
                # greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                if return_log_probs:
                    lp = F.log_softmax(logits, dim=-1)
                    token_log_probs.append(lp[0, next_token.item()].item())
                if next_token.item() == tokenizer.eos_token_id:
                    break
                current_input = next_token
                continue

            # top-p (nucleus) sampling
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if return_log_probs:
                lp = F.log_softmax(logits, dim=-1)
                token_log_probs.append(lp[0, next_token.item()].item())

            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

            # for KV-cache: only pass the new token as input
            current_input = next_token

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if return_log_probs:
        return text, token_log_probs
    return text


def generate_n(
    model,
    tokenizer,
    prompt,
    n_samples=1,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    device="cuda",
):
    """Generate N independent samples for the same prompt."""
    results = []
    for _ in range(n_samples):
        text = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        results.append(text)
    return results
