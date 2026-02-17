"""
Synthetic Math Dataset Generator
- Generates arithmetic problems (addition, subtraction, multiplication, division)
- Clean integer division only
- Consistent output format for SFT training
"""

import random
import json
import argparse


def generate_math_dataset(n_per_op=500, max_num=500, seed=42):
    """Generate synthetic arithmetic dataset with 4 operations."""
    random.seed(seed)
    dataset = []

    # addition
    for _ in range(n_per_op):
        a, b = random.randint(1, max_num), random.randint(1, max_num)
        dataset.append({
            "prompt": f"What is {a} + {b}?",
            "answer": f"The answer is {a + b}",
            "ground_truth": a + b,
        })

    # subtraction (ensure non-negative result)
    for _ in range(n_per_op):
        a, b = random.randint(1, max_num), random.randint(1, max_num)
        if a < b:
            a, b = b, a
        dataset.append({
            "prompt": f"What is {a} - {b}?",
            "answer": f"The answer is {a - b}",
            "ground_truth": a - b,
        })

    # multiplication
    for _ in range(n_per_op):
        a, b = random.randint(1, max_num), random.randint(1, max_num)
        dataset.append({
            "prompt": f"What is {a} * {b}?",
            "answer": f"The answer is {a * b}",
            "ground_truth": a * b,
        })

    # division (clean integer division)
    for _ in range(n_per_op):
        divisor = random.randint(1, 50)
        result = random.randint(1, 50)
        dividend = divisor * result
        dataset.append({
            "prompt": f"What is {dividend} / {divisor}?",
            "answer": f"The answer is {result}",
            "ground_truth": result,
        })

    random.shuffle(dataset)
    return dataset


def generate_gsm8k_format_data(n_samples=500, seed=42):
    """
    Generate word problems in GSM8K-like format with step-by-step solutions.
    Uses templates to create varied arithmetic word problems.
    """
    random.seed(seed)
    dataset = []

    templates = [
        {
            "template": "{name} has {a} apples. {name2} gives {name} {b} more apples. How many apples does {name} have now?",
            "solution": "{name} started with {a} apples.\n{name2} gave {a_name} {b} more.\n{a} + {b} = {answer}\n#### {answer}",
            "op": lambda a, b: a + b,
        },
        {
            "template": "A store has {a} shirts. They sell {b} shirts. How many shirts are left?",
            "solution": "The store started with {a} shirts.\nThey sold {b} shirts.\n{a} - {b} = {answer}\n#### {answer}",
            "op": lambda a, b: a - b,
        },
        {
            "template": "There are {a} boxes with {b} books in each box. How many books are there in total?",
            "solution": "There are {a} boxes.\nEach box has {b} books.\n{a} * {b} = {answer}\n#### {answer}",
            "op": lambda a, b: a * b,
        },
        {
            "template": "{a} students need to be split into {b} equal groups. How many students are in each group?",
            "solution": "There are {a} students total.\nThey need to be split into {b} groups.\n{a} / {b} = {answer}\n#### {answer}",
            "op": lambda a, b: a // b,
        },
        {
            "template": "{name} earns ${a} per hour and works {b} hours. How much does {name} earn?",
            "solution": "{name} earns ${a} per hour.\n{name} works {b} hours.\n{a} * {b} = {answer}\n#### {answer}",
            "op": lambda a, b: a * b,
        },
        {
            "template": "A farmer has {a} chickens. Each chicken lays {b} eggs per day. How many eggs does the farmer get per day?",
            "solution": "The farmer has {a} chickens.\nEach chicken lays {b} eggs.\n{a} * {b} = {answer}\n#### {answer}",
            "op": lambda a, b: a * b,
        },
        {
            "template": "{name} has ${a}. {name} spends ${b} on lunch. How much money does {name} have left?",
            "solution": "{name} started with ${a}.\n{name} spent ${b} on lunch.\n{a} - {b} = {answer}\n#### {answer}",
            "op": lambda a, b: a - b,
        },
        {
            "template": "A bus can carry {b} passengers. If there are {a} passengers waiting, how many buses are needed?",
            "solution": "There are {a} passengers.\nEach bus carries {b} passengers.\n{a} / {b} = {answer} (rounding up)\n#### {answer}",
            "op": lambda a, b: -(-a // b),  # ceiling division
        },
    ]

    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    names2 = ["Tom", "Sarah", "Mike", "Lisa", "James", "Emma", "David", "Nina"]

    for _ in range(n_samples):
        tmpl = random.choice(templates)
        name = random.choice(names)
        name2 = random.choice(names2)

        # pick numbers appropriate for the operation
        if tmpl["op"].__code__.co_code == templates[3]["op"].__code__.co_code:
            # division: ensure clean division
            b = random.randint(2, 20)
            result = random.randint(1, 50)
            a = b * result
        elif tmpl["op"].__code__.co_code == templates[1]["op"].__code__.co_code or \
             tmpl["op"].__code__.co_code == templates[6]["op"].__code__.co_code:
            # subtraction: ensure a >= b
            a = random.randint(10, 500)
            b = random.randint(1, a)
        else:
            a = random.randint(2, 100)
            b = random.randint(2, 50)

        answer = tmpl["op"](a, b)

        question = tmpl["template"].format(name=name, name2=name2, a=a, b=b)
        solution = tmpl["solution"].format(
            name=name, name2=name2, a=a, b=b,
            a_name=name, answer=answer,
        )

        dataset.append({
            "question": question,
            "solution": solution,
            "answer": str(answer),
        })

    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["arithmetic", "word"], default="arithmetic")
    parser.add_argument("--n_per_op", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/train.jsonl")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.type == "arithmetic":
        data = generate_math_dataset(n_per_op=args.n_per_op)
    else:
        data = generate_gsm8k_format_data(n_samples=args.n_per_op * 4)

    with open(args.output, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {len(data)} examples → {args.output}")
