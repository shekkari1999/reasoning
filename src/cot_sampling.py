"""
Inference-Time Scaling
- Chain-of-thought prompting
- Self-consistency via majority voting (N=10)
- Systematic temperature analysis
"""

from collections import Counter
from tqdm import tqdm

from src.inference import generate
from eval.answer_extraction import extract_answer
from eval.math_verifier import verify_answer


COT_PROMPT_TEMPLATE = """{question}

Let's think step by step to solve this problem. Show your reasoning, then put your final answer after "#### "."""


def generate_cot_response(model, tokenizer, question, temperature=0.7, max_new_tokens=512, device="cuda"):
    """Generate a single chain-of-thought response."""
    prompt = COT_PROMPT_TEMPLATE.format(question=question)
    response = generate(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        device=device,
    )
    return response


def self_consistency(
    model,
    tokenizer,
    question,
    n_samples=10,
    temperature=0.7,
    max_new_tokens=512,
    device="cuda",
):
    """
    Self-consistency: generate N chain-of-thought responses and take majority vote.

    Steps:
    1. Sample N diverse reasoning paths (using temperature > 0)
    2. Extract the final answer from each path
    3. Take majority vote across all extracted answers

    Returns:
        best_answer: the most common answer (or None if no answers extracted)
        responses: list of all generated responses
        vote_counts: Counter of answer frequencies
    """
    answers = []
    responses = []

    for _ in range(n_samples):
        response = generate_cot_response(
            model, tokenizer, question,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        responses.append(response)
        answer = extract_answer(response)
        if answer is not None:
            answers.append(str(answer))

    if not answers:
        return None, responses, Counter()

    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0]

    return best_answer, responses, vote_counts


def evaluate_self_consistency(
    model,
    tokenizer,
    problems,
    n_samples=10,
    temperature=0.7,
    device="cuda",
    verbose=True,
):
    """
    Evaluate self-consistency on a dataset of problems.
    Each problem should have 'question' and 'answer' keys.
    """
    correct = 0
    results = []

    iterator = tqdm(problems, desc="Self-consistency eval") if verbose else problems

    for problem in iterator:
        best_answer, responses, votes = self_consistency(
            model, tokenizer, problem["question"],
            n_samples=n_samples, temperature=temperature, device=device,
        )

        is_correct = verify_answer(str(best_answer), str(problem["answer"])) if best_answer else False
        if is_correct:
            correct += 1

        results.append({
            "question": problem["question"],
            "expected": problem["answer"],
            "predicted": best_answer,
            "correct": is_correct,
            "votes": dict(votes),
            "n_responses": len(responses),
        })

    accuracy = correct / len(problems) if problems else 0
    if verbose:
        print(f"\nSelf-consistency accuracy: {correct}/{len(problems)} = {accuracy:.2%}")

    return accuracy, results


def temperature_sweep(
    model,
    tokenizer,
    problems,
    temperatures=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    n_samples=10,
    device="cuda",
):
    """
    Run self-consistency at multiple temperatures to find the optimal setting.
    Returns dict mapping temperature -> accuracy.
    """
    results = {}

    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        accuracy, _ = evaluate_self_consistency(
            model, tokenizer, problems,
            n_samples=n_samples, temperature=temp, device=device,
        )
        results[temp] = accuracy

    print("\n--- Temperature Sweep Summary ---")
    for temp, acc in sorted(results.items()):
        bar = "#" * int(acc * 50)
        print(f"  T={temp:.1f}: {acc:.2%} {bar}")

    best_temp = max(results, key=results.get)
    print(f"\nBest temperature: {best_temp} ({results[best_temp]:.2%})")

    return results
