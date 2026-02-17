"""
Benchmark Runner
- GSM8K and MATH evaluation
- Accuracy tracking with detailed per-example results
- JSON output for analysis
"""

import json
from tqdm import tqdm
from datasets import load_dataset

from eval.math_verifier import verify_answer
from eval.answer_extraction import extract_answer, extract_gsm8k_answer
from src.inference import generate


def load_gsm8k(split="test", max_samples=None):
    """
    Load GSM8K dataset.
    Returns list of dicts with 'question' and 'answer' keys.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    problems = []

    for example in ds:
        # GSM8K answers are after ####
        answer = extract_gsm8k_answer(example["answer"])
        if answer is None:
            answer = example["answer"].split("####")[-1].strip()

        problems.append({
            "question": example["question"],
            "answer": answer,
            "full_solution": example["answer"],
        })

    if max_samples:
        problems = problems[:max_samples]

    print(f"Loaded {len(problems)} problems from GSM8K ({split})")
    return problems


def load_math(split="test", max_samples=None):
    """
    Load MATH dataset.
    Returns list of dicts with 'question', 'answer', 'level', and 'type' keys.
    """
    ds = load_dataset("hendrycks/competition_math", split=split)
    problems = []

    for example in ds:
        problems.append({
            "question": example["problem"],
            "answer": example["solution"],
            "level": example["level"],
            "type": example["type"],
        })

    if max_samples:
        problems = problems[:max_samples]

    print(f"Loaded {len(problems)} problems from MATH ({split})")
    return problems


def evaluate(
    model,
    tokenizer,
    problems,
    max_new_tokens=512,
    temperature=0.0,
    device="cuda",
    verbose=True,
):
    """
    Run evaluation on a list of problems.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        problems: list of dicts with 'question' and 'answer' keys
        max_new_tokens: max generation length
        temperature: 0 = greedy (deterministic)
        device: cuda or cpu
        verbose: print progress

    Returns:
        accuracy (float), results (list of dicts)
    """
    correct = 0
    results = []

    iterator = tqdm(enumerate(problems), total=len(problems), desc="Evaluating") if verbose else enumerate(problems)

    for i, problem in iterator:
        prompt = f"{problem['question']}\nThink step by step. Put your final answer after \"#### \"."

        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )

        predicted = extract_answer(response)
        is_correct = verify_answer(str(predicted), str(problem["answer"])) if predicted else False

        if is_correct:
            correct += 1

        results.append({
            "question": problem["question"],
            "expected": problem["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "full_response": response,
        })

    accuracy = correct / len(problems) if problems else 0

    if verbose:
        print(f"\nAccuracy: {correct}/{len(problems)} = {accuracy:.2%}")

    return accuracy, results


def save_results(results, accuracy, output_path):
    """Save evaluation results to JSON."""
    output = {
        "accuracy": accuracy,
        "n_correct": sum(1 for r in results if r["correct"]),
        "n_total": len(results),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")
