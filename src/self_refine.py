"""
Self-Refinement
- Iterative generate-score-critique loop
- Token log-probability confidence scoring
- Refinement until confidence threshold or max iterations
"""

import torch
import torch.nn.functional as F
from src.inference import generate


def compute_log_prob_score(model, tokenizer, prompt, response, device="cuda"):
    """
    Compute average token log-probability of the response given the prompt.
    Higher score = model is more confident in the response.
    """
    full_text = prompt + " " + response
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    if full_ids.shape[1] <= prompt_len:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    # log probs for response tokens: logits at [prompt_len-1 : -1] predict tokens at [prompt_len : end]
    response_logits = logits[:, prompt_len - 1:-1, :]
    response_ids = full_ids[:, prompt_len:]

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.mean().item()


CRITIQUE_TEMPLATE = """Question: {question}

Previous answer: {response}

The previous answer may contain errors. Review it carefully, then provide a corrected solution.
Think step by step. Put your final answer after "#### "."""


def self_refine(
    model,
    tokenizer,
    question,
    max_iterations=3,
    confidence_threshold=-0.5,
    temperature=0.7,
    device="cuda",
):
    """
    Iteratively generate, score, and refine responses.

    1. Generate initial response
    2. Score confidence via average token log-prob
    3. If below threshold, critique and regenerate
    4. Return the best response (highest confidence)
    """
    # initial generation
    prompt = f"{question}\nThink step by step. Put your final answer after \"#### \"."
    response = generate(
        model, tokenizer, prompt,
        max_new_tokens=512, temperature=temperature, device=device,
    )
    score = compute_log_prob_score(model, tokenizer, prompt, response, device)

    history = [{"response": response, "score": score, "iteration": 0}]

    for i in range(1, max_iterations):
        if score >= confidence_threshold:
            break

        # critique and refine
        critique_prompt = CRITIQUE_TEMPLATE.format(question=question, response=response)
        response = generate(
            model, tokenizer, critique_prompt,
            max_new_tokens=512, temperature=temperature, device=device,
        )
        score = compute_log_prob_score(model, tokenizer, question, response, device)
        history.append({"response": response, "score": score, "iteration": i})

    # return the response with the highest confidence
    best = max(history, key=lambda x: x["score"])
    return best["response"], best["score"], history


def self_refine_batch(model, tokenizer, questions, **kwargs):
    """Run self-refinement on a batch of questions."""
    results = []
    for q in questions:
        response, score, history = self_refine(model, tokenizer, q, **kwargs)
        results.append({
            "question": q,
            "best_response": response,
            "best_score": score,
            "n_iterations": len(history),
            "history": history,
        })
    return results
