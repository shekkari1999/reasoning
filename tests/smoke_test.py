"""Quick checks that run on CPU — no GPU, no model download."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aggregate_results import aggregate_results
from src.data import deterministic_sample_indices
from src.losses import (
    build_completion_mask,
    compute_advantages_dr_grpo,
    dr_grpo_loss,
    get_per_token_logprobs,
)
from src.rewards import (
    compute_reward,
    extract_answer_gsm8k,
    extract_model_answer,
)


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


def test_math_reward_preserves_latex():
    pred = "<answer>\\frac{1}{2}</answer>"
    gt = "\\boxed{0.5}"
    assert compute_reward(pred, gt, "math500") == 1.0


def test_dr_grpo_advantages():
    import torch

    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    adv_dr = compute_advantages_dr_grpo(rewards)
    assert adv_dr.shape == (1, 4)
    assert adv_dr.tolist() == [[0.5, -0.5, -0.5, 0.5]]


def test_completion_mask_excludes_prompt_and_post_eos_padding():
    import torch

    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0]])
    mask = build_completion_mask(attention_mask, prompt_width=3)
    assert mask.tolist() == [[False, False, True, True, False]]


def test_production_dr_grpo_loss():
    from types import SimpleNamespace

    import torch

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 5, 7))

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.logits)

    policy = TinyPolicy()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    old = get_per_token_logprobs(policy, input_ids, attention_mask).detach()
    mask = build_completion_mask(attention_mask, prompt_width=3)
    result = dr_grpo_loss(
        policy_model=policy,
        input_ids=input_ids,
        attention_mask=attention_mask,
        old_per_token_logprobs=old,
        advantages=torch.tensor([1.0]),
        completion_mask=mask,
        max_completion_length=3,
    )
    result["loss"].backward()
    assert policy.logits.grad is not None
    assert abs(result["loss"].item() + 2 / 3) < 1e-6


def test_deterministic_sampling():
    first = deterministic_sample_indices(100, 10, seed=42)
    assert first == deterministic_sample_indices(100, 10, seed=42)
    assert first != deterministic_sample_indices(100, 10, seed=43)


def test_result_aggregation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for stage in ("base", "sft", "dr_grpo"):
            payload = {
                "model": stage,
                "stage": stage,
                "prompt_mode": "sft",
                "dtype": "bf16",
                "max_new_tokens": 512,
                "seed": 42,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "git_commit": "abc123",
                "datasets": {},
                "gsm8k": {"accuracy": 50.0, "correct": 1, "total": 2},
                "math500": {"accuracy": 50.0, "correct": 1, "total": 2},
            }
            (root / f"{stage}_eval_summary.json").write_text(json.dumps(payload))
            detailed = [{"correct": True}, {"correct": False}]
            for dataset in ("gsm8k", "math500"):
                (root / f"{stage}_{dataset}_detailed.json").write_text(
                    json.dumps(detailed)
                )
        summary = aggregate_results(root)
        assert summary["provenance"] == "generated_from_evaluator_summaries"
        assert summary["stages"]["sft"]["datasets"]["gsm8k"]["correct"] == 1


def test_imports():
    import yaml

    for cfg in (ROOT / "configs").glob("*.yaml"):
        yaml.safe_load(cfg.read_text())


if __name__ == "__main__":
    tests = [
        test_gsm8k_extraction,
        test_reward_correct,
        test_math_reward_preserves_latex,
        test_dr_grpo_advantages,
        test_completion_mask_excludes_prompt_and_post_eos_padding,
        test_production_dr_grpo_loss,
        test_deterministic_sampling,
        test_result_aggregation,
        test_imports,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print("\nAll smoke tests passed.")
