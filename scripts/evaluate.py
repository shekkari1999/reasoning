"""
Evaluation Script
- Run benchmarks on any checkpoint
- Supports GSM8K and MATH
- Multiple evaluation modes: greedy, self-consistency, self-refine
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark import load_gsm8k, load_math, evaluate, save_results
from src.cot_sampling import evaluate_self_consistency, temperature_sweep


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on math benchmarks")
    parser.add_argument("--model", type=str, required=True, help="HF model name or checkpoint path")
    parser.add_argument("--benchmark", type=str, choices=["gsm8k", "math"], default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--mode", type=str, choices=["greedy", "self_consistency", "temp_sweep"], default="greedy")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples for self-consistency")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    # load dataset
    if args.benchmark == "gsm8k":
        problems = load_gsm8k(args.split, max_samples=args.max_samples)
    else:
        problems = load_math(args.split, max_samples=args.max_samples)

    print(f"Evaluating {args.model} on {args.benchmark} ({len(problems)} problems)")
    print(f"Mode: {args.mode}")

    if args.mode == "greedy":
        accuracy, results = evaluate(
            model, tokenizer, problems,
            temperature=args.temperature,
            device=args.device,
            verbose=True,
        )

    elif args.mode == "self_consistency":
        accuracy, results = evaluate_self_consistency(
            model, tokenizer, problems,
            n_samples=args.n_samples,
            temperature=args.temperature if args.temperature > 0 else 0.7,
            device=args.device,
        )

    elif args.mode == "temp_sweep":
        results = temperature_sweep(
            model, tokenizer, problems,
            n_samples=args.n_samples,
            device=args.device,
        )
        accuracy = max(results.values())

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_results(results, accuracy, args.output)


if __name__ == "__main__":
    main()
