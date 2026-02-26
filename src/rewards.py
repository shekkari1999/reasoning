"""
Reward functions and answer extraction utilities.
Shared between eval (eval.py) and RL training (grpo/dr_grpo/dapo).

Handles two answer formats:
  - GSM8K:   "#### <number>"
  - MATH500: "\\boxed{<answer>}"
  - Model:   "<answer>...</answer>" (after SFT)
"""

import re
import math
from typing import Optional


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K ground-truth format: #### <number>"""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return normalize_numeric(match.group(1).strip())
    return None


def extract_answer_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} (MATH benchmark format).
    Handles nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return normalize_answer(text[start:i].strip())
            depth -= 1
    return None


def extract_answer_tags(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags (our SFT/RL format)."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return normalize_answer(match.group(1).strip())
    return None


def extract_model_answer(text: str) -> Optional[str]:
    """Try all extraction methods in priority order."""
    answer = extract_answer_tags(text)
    if answer is not None:
        return answer

    answer = extract_answer_boxed(text)
    if answer is not None:
        return answer

    answer = extract_answer_gsm8k(text)
    if answer is not None:
        return answer

    return extract_last_number(text)


def extract_last_number(text: str) -> Optional[str]:
    """Fallback: extract the last standalone number from text."""
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    if matches:
        return normalize_numeric(matches[-1])
    return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """General normalization: strip LaTeX formatting, whitespace, etc."""
    text = text.strip()
    text = text.replace("\\$", "").replace("$", "")
    text = text.replace("\\%", "%")
    text = text.replace("\\text{", "").replace("\\mathrm{", "")
    text = text.replace("\\frac", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\ ", "")
    text = text.rstrip(".")
    text = text.strip()

    numeric = normalize_numeric(text)
    if numeric is not None:
        return numeric

    return text.lower()


def normalize_numeric(text: str) -> Optional[str]:
    """Normalize a numeric string: remove commas, evaluate fractions, round."""
    text = text.strip().replace(",", "").replace(" ", "")

    frac_match = re.match(r"^(-?\d+)/(\d+)$", text)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            val = num / den
            if val == int(val):
                return str(int(val))
            return str(round(val, 6))

    try:
        val = float(text)
        if math.isnan(val) or math.isinf(val):
            return None
        if val == int(val) and "." not in text:
            return str(int(val))
        return str(round(val, 6))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Reward computation (used in RL training)
# ---------------------------------------------------------------------------

def compute_reward(completion: str, ground_truth: str, dataset: str = "gsm8k") -> float:
    """Binary exact-match reward.

    Args:
        completion: Model-generated text
        ground_truth: Raw ground truth string from the dataset
        dataset: "gsm8k" or "math500"

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    pred = extract_model_answer(completion)
    if pred is None:
        return 0.0

    if dataset == "gsm8k":
        gt = extract_answer_gsm8k(ground_truth)
    elif dataset in ("math500", "math"):
        gt = extract_answer_boxed(ground_truth)
    else:
        gt = normalize_answer(ground_truth)

    if gt is None:
        return 0.0

    return 1.0 if pred == gt else 0.0


def compute_batch_rewards(
    completions: list[str],
    ground_truths: list[str],
    dataset: str = "gsm8k"
) -> list[float]:
    """Compute rewards for a batch of completions."""
    return [
        compute_reward(c, gt, dataset)
        for c, gt in zip(completions, ground_truths)
    ]


# ---------------------------------------------------------------------------
# DAPO-specific: overlong penalty
# ---------------------------------------------------------------------------

def compute_reward_with_overlong_penalty(
    completion: str,
    ground_truth: str,
    completion_len: int,
    max_len: int,
    dataset: str = "gsm8k",
    penalty_scale: float = -0.5
) -> float:
    """Reward with DAPO overlong penalty.

    If the completion hits max_len without producing a valid answer,
    apply a soft penalty to discourage rambling.
    """
    base_reward = compute_reward(completion, ground_truth, dataset)

    if base_reward == 0.0 and completion_len >= max_len - 1:
        return penalty_scale

    return base_reward
