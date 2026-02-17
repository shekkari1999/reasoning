"""
SFT Training Script
- Fine-tune base model on math question-answer pairs
- Supports full fine-tuning and LoRA
- Masked loss on completion tokens only
"""

import os
import sys
import argparse
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_math_data import generate_math_dataset, generate_gsm8k_format_data
from eval.benchmark import load_gsm8k, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--data_type", type=str, choices=["arithmetic", "word", "both"], default="both")
    parser.add_argument("--n_per_op", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--eval_samples", type=int, default=50, help="N samples for eval during training")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def tokenize_example(tokenizer, prompt, answer, device):
    """
    Tokenize a prompt-answer pair with masked labels for prompt tokens.

    Returns input_ids and labels as tensors with batch dimension.
    Labels have -100 for prompt tokens (ignored by cross-entropy loss).
    """
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(" " + answer)
    input_ids = prompt_ids + answer_ids

    # mask prompt tokens with -100, keep answer tokens
    labels = [-100] * len(prompt_ids) + answer_ids

    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    labels = torch.tensor(labels, device=device).unsqueeze(0)

    return input_ids, labels


def build_dataset(data_type, n_per_op):
    """Build SFT training dataset."""
    dataset = []

    if data_type in ("arithmetic", "both"):
        arith_data = generate_math_dataset(n_per_op=n_per_op)
        # convert to common format with 'question' key
        for d in arith_data:
            dataset.append({
                "question": d["prompt"],
                "answer": d["answer"],
            })

    if data_type in ("word", "both"):
        word_data = generate_gsm8k_format_data(n_samples=n_per_op * 2)
        for d in word_data:
            # for SFT, the answer is the full step-by-step solution
            dataset.append({
                "question": d["question"],
                "answer": d["solution"],
            })

    import random
    random.shuffle(dataset)
    print(f"Built SFT dataset: {len(dataset)} examples ({data_type})")
    return dataset


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    dataset = build_dataset(args.data_type, args.n_per_op)

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, example in pbar:
            input_ids, labels = tokenize_example(
                tokenizer, example["question"], example["answer"], args.device,
            )

            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / (i+1):.4f}"})

            # periodic evaluation
            if args.eval_every > 0 and global_step % args.eval_every == 0:
                model.eval()
                print(f"\n--- Eval at step {global_step} ---")
                eval_problems = load_gsm8k("test", max_samples=args.eval_samples)
                accuracy, _ = evaluate(model, tokenizer, eval_problems, device=args.device)
                print(f"Step {global_step} | Eval accuracy: {accuracy:.2%}")
                model.train()

        avg_loss = total_loss / len(dataset)
        print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

    # save final model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved SFT model to {args.output_dir}")

    # final evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    eval_problems = load_gsm8k("test", max_samples=100)
    accuracy, results = evaluate(model, tokenizer, eval_problems, device=args.device)
    print(f"Final accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
