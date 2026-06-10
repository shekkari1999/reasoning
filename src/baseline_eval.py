"""
Baseline Evaluation — Qwen-2.5-3B on GSM8K and MATH500.

Establishes baseline accuracy before any fine-tuning.
Run on a single GPU (Colab, RunPod, etc.)

Usage:
    python src/baseline_eval.py --model Qwen/Qwen2.5-3B --dataset both
    python src/baseline_eval.py --model checkpoints/sft/step_300 --dataset both --prompt_mode sft
"""

import json
import sys
import time
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rewards import (
    extract_answer_boxed,
    extract_answer_gsm8k,
    extract_model_answer,
    normalize_answer,
)
from src.profiling_utils import get_visible_gpu_memory, log_memory, reset_peak_memory


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

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


def format_prompt_base(question: str, dataset: str) -> str:
    if dataset == "gsm8k":
        shots = ""
        for ex in GSM8K_FEWSHOT:
            shots += f"Question: {ex['q']}\nAnswer: {ex['a']}\n\n"
        return f"{shots}Question: {question}\nAnswer:"
    else:
        shots = ""
        for ex in MATH_FEWSHOT:
            shots += f"Problem: {ex['q']}\nSolution: {ex['a']}\n\n"
        return f"{shots}Problem: {question}\nSolution:"


def format_prompt_sft(question: str, dataset: str) -> str:
    return (
        f"You are a helpful assistant that solves math problems step by step. "
        f"Show your reasoning inside <think>...</think> tags, then give your "
        f"final answer inside <answer>...</answer> tags.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens=512):
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=2048,
    ).to(model.device)

    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )

    completions = []
    for i, out in enumerate(output_ids):
        prompt_len = inputs["input_ids"][i].shape[0]
        text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for stop in ["\n\nQuestion:", "\n\nProblem:", "\n\n\n"]:
            if stop in text:
                text = text[:text.index(stop)]
        completions.append(text)
    return completions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, examples, dataset, prompt_mode="base",
             batch_size=16, max_new_tokens=512):
    format_fn = format_prompt_sft if prompt_mode == "sft" else format_prompt_base
    results = []
    correct = 0
    total = len(examples)
    t_start = time.time()

    for i in range(0, total, batch_size):
        batch = examples[i : i + batch_size]
        prompts = [format_fn(ex["question"], dataset) for ex in batch]
        completions = generate_batch(model, tokenizer, prompts, max_new_tokens)

        for ex, completion in zip(batch, completions):
            pred = extract_model_answer(completion)
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
                "completion": completion[:500],
            })

        done = min(i + batch_size, total)
        elapsed = time.time() - t_start
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done:>4}/{total}] Acc: {correct/done*100:5.1f}% | "
              f"{rate:.1f} samples/s | ETA: {eta:.0f}s")

    elapsed = time.time() - t_start
    return {
        "dataset": dataset,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2),
        "elapsed_seconds": round(elapsed, 1),
        "samples_per_second": round(total / elapsed, 2),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["gsm8k", "math500", "both"])
    parser.add_argument("--prompt_mode", type=str, default="sft",
                        choices=["base", "sft"],
                        help="sft = zero-shot CoT prompt (fair ladder). base = few-shot (legacy only).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--stage", type=str, default="base",
                        help="Label: base, sft, grpo, dr_grpo, dapo")
    args = parser.parse_args()

    # Load model
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"Loading {args.model} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    log_memory("after model load")
    reset_peak_memory()

    # Load datasets
    datasets_to_eval = []
    if args.dataset in ("gsm8k", "both"):
        raw = load_dataset("openai/gsm8k", "main", split="test")
        data = [{"question": r["question"], "raw_answer": r["answer"],
                 "dataset": "gsm8k"} for r in raw]
        if args.max_samples > 0:
            data = data[:args.max_samples]
        datasets_to_eval.append(("gsm8k", data))

    if args.dataset in ("math500", "both"):
        raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
        data = [{"question": r["problem"], "raw_answer": r["answer"],
                 "dataset": "math500"} for r in raw]
        if args.max_samples > 0:
            data = data[:args.max_samples]
        datasets_to_eval.append(("math500", data))

    # Evaluate
    all_results = {}
    for ds_name, examples in datasets_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name.upper()} ({len(examples)} samples)")
        print(f"{'='*60}")

        result = evaluate(model, tokenizer, examples, ds_name,
                         prompt_mode=args.prompt_mode,
                         batch_size=args.batch_size,
                         max_new_tokens=args.max_new_tokens)

        print(f"\n{ds_name.upper()}: {result['accuracy']}% "
              f"({result['correct']}/{result['total']})")
        all_results[ds_name] = result

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    memory = get_visible_gpu_memory()
    save_data = {
        "model": args.model,
        "stage": args.stage,
        "prompt_mode": args.prompt_mode,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "memory": {
            "peak_gb_per_gpu": [g["peak_gb"] for g in memory],
            "peak_gb_max": max((g["peak_gb"] for g in memory), default=0.0),
            "gpus": memory,
        },
    }
    log_memory("after eval")
    for ds_name, result in all_results.items():
        save_data[ds_name] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        # Save detailed results
        with open(output_dir / f"{args.stage}_{ds_name}_detailed.json", "w") as f:
            json.dump(result["results"], f, indent=2)

    with open(output_dir / f"{args.stage}_eval_summary.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model} ({args.stage})")
    print(f"{'='*60}")
    for ds_name, result in all_results.items():
        print(f"  {ds_name.upper():>8}: {result['accuracy']:5.1f}%  "
              f"({result['correct']}/{result['total']})")
    print(f"{'='*60}")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
