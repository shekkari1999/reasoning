"""Quick checks that run on CPU — no GPU, no model download."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rewards import (
    compute_reward,
    extract_answer_gsm8k,
    extract_model_answer,
)
from src.losses import compute_advantages_dr_grpo


def test_gsm8k_extraction():
    gt = "Some reasoning here.\n#### 42"
    assert extract_answer_gsm8k(gt) == "42"

    pred = "<think>add 1+1</think>\n<answer>2</answer>"
    assert extract_model_answer(pred) == "2"
    assert compute_reward(pred, gt, "gsm8k") == 0.0


def test_reward_correct():
    gt = "Reasoning.\n#### 18"
    pred = "Let me think...\n<answer>18</answer>"
    assert compute_reward(pred, gt, "gsm8k") == 1.0


def test_dr_grpo_advantages():
    import torch

    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    adv_dr = compute_advantages_dr_grpo(rewards)
    assert adv_dr.shape == (1, 4)
    assert adv_dr.tolist() == [[0.5, -0.5, -0.5, 0.5]]


def test_clipped_surrogate():
    import torch

    old_lp = torch.tensor([0.0, -1.0, -2.0, -0.5])
    new_lp = torch.tensor([-0.1, -0.8, -2.5, -0.3], requires_grad=True)
    adv = torch.tensor([1.0, -1.0, -1.0, 1.0])
    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    loss = -torch.min(ratio * adv, clipped * adv).mean()
    loss.backward()
    assert new_lp.grad is not None


def test_imports():
    import yaml

    for cfg in (ROOT / "configs").glob("*.yaml"):
        yaml.safe_load(cfg.read_text())


if __name__ == "__main__":
    tests = [
        test_gsm8k_extraction,
        test_reward_correct,
        test_dr_grpo_advantages,
        test_clipped_surrogate,
        test_imports,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print("\nAll smoke tests passed.")
