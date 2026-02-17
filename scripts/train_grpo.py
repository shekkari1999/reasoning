"""
GRPO Training Script
- Load SFT model as starting point
- Train with group-relative policy optimization
- Binary verifiable rewards on math problems
"""

import os
import sys
import copy
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grpo import train_grpo
from eval.benchmark import load_gsm8k, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--model", type=str, default="checkpoints/sft", help="Path to SFT checkpoint or HF model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--n_samples", type=int, default=8, help="Number of rollout samples per question")
    parser.add_argument("--kl_coeff", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training data size")
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    # frozen reference model for KL penalty
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # load training data
    print("Loading GSM8K training data...")
    dataset = load_gsm8k(split="train", max_samples=args.max_train_samples)

    # evaluate before training
    print("\n--- Pre-training Evaluation ---")
    model.eval()
    eval_problems = load_gsm8k("test", max_samples=args.eval_samples)
    pre_accuracy, _ = evaluate(model, tokenizer, eval_problems, device=args.device)
    print(f"Pre-GRPO accuracy: {pre_accuracy:.2%}")

    # train
    print("\n--- Starting GRPO Training ---")
    model.train()
    metrics = train_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        kl_coeff=args.kl_coeff,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        checkpoint_dir=args.output_dir,
    )

    # evaluate after training
    print("\n--- Post-training Evaluation ---")
    model.eval()
    post_accuracy, results = evaluate(model, tokenizer, eval_problems, device=args.device)
    print(f"Post-GRPO accuracy: {post_accuracy:.2%}")
    print(f"Improvement: {pre_accuracy:.2%} → {post_accuracy:.2%} ({post_accuracy - pre_accuracy:+.2%})")

    # save final results
    import json
    results_path = os.path.join(args.output_dir, "training_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "pre_accuracy": pre_accuracy,
            "post_accuracy": post_accuracy,
            "epoch_rewards": metrics["epoch_rewards"],
            "epoch_losses": metrics["epoch_losses"],
        }, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
