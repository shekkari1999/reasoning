"""
Evaluation script for GSM8K and MATH500.

Usage (Colab or CLI):
    python src/eval.py \
        --model Qwen/Qwen2.5-3B \
        --dataset gsm8k \
        --max_samples 0 \
        --batch_size 8 \
        --max_new_tokens 512 \
        --output results/base_eval.json

Set --max_samples 0 for full eval, or a small number for quick sanity checks.
Supports: base model, SFT checkpoint, RL checkpoint (auto-detects prompt format).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add parent dir so we can import rewards
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rewards import (
    extract_model_answer,
    extract_answer_gsm8k,
    extract_answer_boxed,
    normalize_answer,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k_test() -> list[dict]:
    """Load GSM8K test split. Returns list of {question, answer, gt_answer}."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for row in ds:
        # Ground truth answer is after "####"
        raw_answer = row["answer"]
        examples.append({
            "question": row["question"],
            "raw_answer": raw_answer,
            "dataset": "gsm8k",
        })
    return examples


def load_math500() -> list[dict]:
    """Load MATH-500 test set. Returns list of {question, answer, gt_answer}."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    examples = []
    for row in ds:
        examples.append({
            "question": row["problem"],
            "raw_answer": row["answer"],
            "dataset": "math500",
        })
    return examples


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

# Few-shot exemplars for base model (no SFT tags)
GSM8K_FEWSHOT = [
    {
        "q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
]

MATH_FEWSHOT = [
    {
        "q": "What is the value of $\\sqrt{36}$?",
        "a": "We need to find a number that, when multiplied by itself, equals 36. Since $6 \\times 6 = 36$, we have $\\sqrt{36} = \\boxed{6}$.",
    },
]


def format_prompt_base(question: str, dataset: str, use_fewshot: bool = True) -> str:
    """Format prompt for base model (no SFT). Uses few-shot to teach answer format."""
    if dataset == "gsm8k":
        fewshot = GSM8K_FEWSHOT if use_fewshot else []
        shots = ""
        for ex in fewshot:
            shots += f"Question: {ex['q']}\nAnswer: {ex['a']}\n\n"
        return f"{shots}Question: {question}\nAnswer:"
    else:  # math500
        fewshot = MATH_FEWSHOT if use_fewshot else []
        shots = ""
        for ex in fewshot:
            shots += f"Problem: {ex['q']}\nSolution: {ex['a']}\n\n"
        return f"{shots}Problem: {question}\nSolution:"


def format_prompt_sft(question: str, dataset: str) -> str:
    """Format prompt for SFT/RL models that expect <think>/<answer> tags."""
    return (
        f"Solve the following problem. Show your reasoning inside <think>...</think> "
        f"tags, then give your final answer inside <answer>...</answer> tags.\n\n"
        f"Problem: {question}"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """Generate completions for a batch of prompts. Greedy decoding by default."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        pad_token_id=tokenizer.pad_token_id,
    )

    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature

    output_ids = model.generate(**generate_kwargs)

    # Decode only the generated tokens (strip the prompt)
    completions = []
    for i, out in enumerate(output_ids):
        prompt_len = inputs["input_ids"][i].shape[0]
        generated = out[prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        # Truncate at stop strings — base model generates fake follow-up Q&A
        for stop in ["\n\nQuestion:", "\n\nProblem:", "\n\n\n"]:
            if stop in text:
                text = text[:text.index(stop)]
        completions.append(text)

    return completions


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    examples: list[dict],
    dataset: str,
    prompt_mode: str = "base",
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> dict:
    """Run evaluation on a list of examples.

    Args:
        model: HF model
        tokenizer: HF tokenizer
        examples: List of {question, raw_answer, dataset}
        dataset: "gsm8k" or "math500"
        prompt_mode: "base" (few-shot) or "sft" (<think>/<answer> tags)
        batch_size: Batch size for generation
        max_new_tokens: Max tokens to generate per problem

    Returns:
        Dict with accuracy, per-example results, timing
    """
    format_fn = format_prompt_sft if prompt_mode == "sft" else format_prompt_base

    results = []
    correct = 0
    total = len(examples)

    t_start = time.time()

    for i in range(0, total, batch_size):
        batch = examples[i : i + batch_size]
        prompts = [format_fn(ex["question"], dataset) for ex in batch]

        completions = generate_batch(
            model, tokenizer, prompts, max_new_tokens=max_new_tokens
        )

        for ex, completion in zip(batch, completions):
            # Extract predicted answer
            pred = extract_model_answer(completion)

            # Extract ground truth
            if dataset == "gsm8k":
                gt = extract_answer_gsm8k(ex["raw_answer"])
            else:
                gt = extract_answer_boxed(ex["raw_answer"])
                if gt is None:
                    gt = normalize_answer(ex["raw_answer"])

            is_correct = pred is not None and gt is not None and pred == gt

            if is_correct:
                correct += 1

            results.append({
                "question": ex["question"],
                "ground_truth": gt,
                "predicted": pred,
                "correct": is_correct,
                "completion": completion[:500],  # truncate for storage
            })

        # Progress
        done = min(i + batch_size, total)
        acc_so_far = correct / done * 100
        print(f"  [{done}/{total}] Running accuracy: {acc_so_far:.1f}%")

    elapsed = time.time() - t_start
    accuracy = correct / total * 100

    return {
        "dataset": dataset,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "samples_per_second": round(total / elapsed, 2),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K / MATH500")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model name or path to local checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gsm8k", "math500", "both"],
                        help="Which benchmark to run")
    parser.add_argument("--prompt_mode", type=str, default="base",
                        choices=["base", "sft"],
                        help="'base' for few-shot, 'sft' for <think>/<answer> format")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit number of samples (0 = full dataset)")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Model dtype (use fp16 for T4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    # ---- Load model ----
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {model.device}, Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ---- Load datasets ----
    datasets_to_eval = []
    if args.dataset in ("gsm8k", "both"):
        datasets_to_eval.append(("gsm8k", load_gsm8k_test()))
    if args.dataset in ("math500", "both"):
        datasets_to_eval.append(("math500", load_math500()))

    # ---- Run eval ----
    all_results = {}
    for ds_name, examples in datasets_to_eval:
        if args.max_samples > 0:
            examples = examples[: args.max_samples]

        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name.upper()} ({len(examples)} samples)")
        print(f"Prompt mode: {args.prompt_mode} | Batch size: {args.batch_size}")
        print(f"{'='*60}")

        result = evaluate(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            dataset=ds_name,
            prompt_mode=args.prompt_mode,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"\n{ds_name.upper()} Results:")
        print(f"  Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")
        print(f"  Time: {result['elapsed_seconds']}s ({result['samples_per_second']} samples/sec)")

        all_results[ds_name] = result

    # ---- Save results ----
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary (without per-example completions for readability)
        summary = {}
        for ds_name, result in all_results.items():
            summary[ds_name] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "elapsed_seconds": result["elapsed_seconds"],
            }

        save_data = {
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "dtype": args.dtype,
            "summary": summary,
            "detailed_results": {
                ds_name: result["results"]
                for ds_name, result in all_results.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # ---- Print final summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*60}")
    for ds_name, result in all_results.items():
        print(f"  {ds_name.upper():>8}: {result['accuracy']:5.1f}%  ({result['correct']}/{result['total']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
