"""
Baseline Analysis — Extended metrics and plots.

Run AFTER baseline_eval.py. Produces:
  - Pass@K analysis (GSM8K + MATH500)
  - Response length distributions
  - MATH500 per-category and per-difficulty breakdown
  - Per-problem consistency analysis
  - All plots saved to results/

Usage:
    python src/baseline_analysis.py --model Qwen/Qwen2.5-3B --pass_k_samples 200
"""

import os
import re
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Optional
from math import comb
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Import extraction functions (same as baseline_eval.py)
# ---------------------------------------------------------------------------

def extract_answer_gsm8k(text):
    match = re.search(r"####\s*(.+)", text)
    return normalize_numeric(match.group(1).strip()) if match else None

def extract_answer_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1: return None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            if depth == 0: return normalize_answer(text[start:i].strip())
            depth -= 1
    return None

def extract_answer_tags(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return normalize_answer(match.group(1).strip()) if match else None

def extract_last_number(text):
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return normalize_numeric(matches[-1]) if matches else None

def extract_model_answer(text):
    for fn in [extract_answer_tags, extract_answer_boxed, extract_answer_gsm8k, extract_last_number]:
        ans = fn(text)
        if ans is not None: return ans
    return None

def normalize_answer(text):
    text = text.strip()
    for r in ["\\$","$","\\%","\\text{","\\mathrm{","\\frac","\\left","\\right","\\,","\\ "]:
        text = text.replace(r, "")
    text = text.replace("%","").rstrip(".").strip()
    numeric = normalize_numeric(text)
    return numeric if numeric is not None else text.lower()

def normalize_numeric(text):
    text = text.strip().replace(",","").replace(" ","")
    frac = re.match(r"^(-?\d+)/(\d+)$", text)
    if frac:
        n, d = int(frac.group(1)), int(frac.group(2))
        if d != 0:
            v = n/d
            return str(int(v)) if v == int(v) else str(round(v,6))
    try:
        v = float(text)
        if math.isnan(v) or math.isinf(v): return None
        if v == int(v) and "." not in text: return str(int(v))
        return str(round(v,6))
    except ValueError: return None


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

def setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Generation with sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch_greedy(model, tokenizer, prompts, max_new_tokens=512):
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
    completions = []
    for i, out in enumerate(output_ids):
        prompt_len = inputs["input_ids"][i].shape[0]
        text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for stop in ["\n\nQuestion:", "\n\nProblem:", "\n\n\n"]:
            if stop in text: text = text[:text.index(stop)]
        completions.append(text)
    return completions


@torch.no_grad()
def generate_batch_sampled(model, tokenizer, prompts, max_new_tokens=512,
                           temperature=0.7, num_samples=8):
    """Generate num_samples completions per prompt in a single batched call."""
    tokenizer.padding_side = "left"
    expanded = [p for p in prompts for _ in range(num_samples)]
    inputs = tokenizer(expanded, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=temperature,
                                top_p=0.95, pad_token_id=tokenizer.pad_token_id)
    all_samples = [[] for _ in prompts]
    for i, out in enumerate(output_ids):
        prompt_idx = i // num_samples
        prompt_len = inputs["input_ids"][i].shape[0]
        text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for stop in ["\n\nQuestion:", "\n\nProblem:", "\n\n\n"]:
            if stop in text: text = text[:text.index(stop)]
        all_samples[prompt_idx].append(text)
    return all_samples


# ---------------------------------------------------------------------------
# Pass@K
# ---------------------------------------------------------------------------

def compute_pass_at_k(per_problem_samples, ground_truths, k_values, dataset):
    results = {k: 0.0 for k in k_values}
    n = len(per_problem_samples[0])
    valid = 0

    for samples, gt_raw in zip(per_problem_samples, ground_truths):
        if dataset == "gsm8k":
            gt = extract_answer_gsm8k(gt_raw)
        else:
            gt = extract_answer_boxed(gt_raw)
            if gt is None:
                gt = normalize_answer(gt_raw)
        if gt is None:
            continue

        valid += 1
        c = sum(1 for s in samples if extract_model_answer(s) == gt)

        for k in k_values:
            if k > n: continue
            if c == 0:
                pass_k = 0.0
            elif n - c < k:
                pass_k = 1.0
            else:
                pass_k = 1.0 - comb(n - c, k) / comb(n, k)
            results[k] += pass_k

    return {k: v / valid * 100 for k, v in results.items()} if valid > 0 else results


def run_pass_at_k(model, tokenizer, data, dataset, K=8, temperature=0.7,
                  n_samples=200, batch_size=8, prompt_fn=None):
    """Run pass@K analysis on a subset."""
    random.seed(42)
    subset = random.sample(data, min(n_samples, len(data)))

    if prompt_fn is None:
        from functools import partial
        prompt_fn = lambda q: (
            format_prompt_base_gsm8k(q) if dataset == "gsm8k"
            else format_prompt_base_math(q)
        )

    per_problem_samples = []
    t_start = time.time()

    for i in range(0, len(subset), batch_size):
        batch = subset[i : i + batch_size]
        prompts = [format_prompt_base(ex["question"], dataset) for ex in batch]
        samples = generate_batch_sampled(
            model, tokenizer, prompts,
            max_new_tokens=512, temperature=temperature, num_samples=K,
        )
        per_problem_samples.extend(samples)

        done = min(i + batch_size, len(subset))
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(subset) - done) / rate if rate > 0 else 0
        print(f"  [{done:>4}/{len(subset)}] {rate:.1f} problems/s | ETA: {eta:.0f}s")

    ground_truths = [ex["raw_answer"] for ex in subset]
    k_values = [1, 2, 4, K]
    pass_at_k = compute_pass_at_k(per_problem_samples, ground_truths, k_values, dataset)

    return pass_at_k, per_problem_samples, subset, ground_truths


def format_prompt_base(question, dataset):
    GSM8K_FEWSHOT = [
        {"q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
         "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."},
        {"q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
         "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."},
    ]
    MATH_FEWSHOT = [
        {"q": "What is the value of $\\sqrt{36}$?",
         "a": "We need to find a number that, when multiplied by itself, equals 36. Since $6 \\times 6 = 36$, we have $\\sqrt{36} = \\boxed{6}$."},
    ]
    if dataset == "gsm8k":
        shots = "".join(f"Question: {e['q']}\nAnswer: {e['a']}\n\n" for e in GSM8K_FEWSHOT)
        return f"{shots}Question: {question}\nAnswer:"
    else:
        shots = "".join(f"Problem: {e['q']}\nSolution: {e['a']}\n\n" for e in MATH_FEWSHOT)
        return f"{shots}Problem: {question}\nSolution:"


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------

def compute_consistency(per_problem_samples, ground_truths, dataset, K):
    correct_counts = []
    for samples, gt_raw in zip(per_problem_samples, ground_truths):
        if dataset == "gsm8k":
            gt = extract_answer_gsm8k(gt_raw)
        else:
            gt = extract_answer_boxed(gt_raw)
            if gt is None: gt = normalize_answer(gt_raw)
        if gt is None:
            correct_counts.append(0)
            continue
        c = sum(1 for s in samples if extract_model_answer(s) == gt)
        correct_counts.append(c)

    always_correct = sum(1 for c in correct_counts if c == K)
    always_wrong = sum(1 for c in correct_counts if c == 0)
    mixed = sum(1 for c in correct_counts if 0 < c < K)
    return correct_counts, always_correct, mixed, always_wrong


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pass_at_k(gsm8k_pak, math500_pak, K, temperature, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, pak, color, title in [
        (ax1, gsm8k_pak, COLORS[0], "GSM8K"),
        (ax2, math500_pak, COLORS[1], "MATH500"),
    ]:
        ks = list(pak.keys())
        accs = list(pak.values())
        ax.plot(ks, accs, "o-", color=color, linewidth=2.5, markersize=10,
                markeredgecolor="black", markeredgewidth=0.5)
        for k, acc in zip(ks, accs):
            ax.annotate(f"{acc:.1f}%", (k, acc), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontweight="bold", fontsize=11)
        gap = pak[K] - pak[1]
        ax.fill_between([ks[0], ks[-1]], pak[1], pak[K], alpha=0.1, color=color)
        ax.annotate(f"Gap: {gap:.1f}%", xy=((ks[0]+ks[-1])/2, (pak[1]+pak[K])/2),
                    ha="center", fontsize=10, fontstyle="italic", color=color)
        ax.set_xlabel("K")
        ax.set_ylabel("Pass@K (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(ks)
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Qwen-2.5-3B Base — Pass@K (temp={temperature})",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_pass_at_k.png", bbox_inches="tight")
    plt.close()
    print("Saved: baseline_pass_at_k.png")


def plot_consistency(gsm8k_data, math500_data, K, output_dir):
    gsm8k_counts, gsm8k_ac, gsm8k_mix, gsm8k_aw = gsm8k_data
    math500_counts, math500_ac, math500_mix, math500_aw = math500_data

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row, (counts, ac, mix, aw, color, name) in enumerate([
        (gsm8k_counts, gsm8k_ac, gsm8k_mix, gsm8k_aw, COLORS[0], "GSM8K"),
        (math500_counts, math500_ac, math500_mix, math500_aw, COLORS[1], "MATH500"),
    ]):
        axes[row][0].hist(counts, bins=range(K+2), color=color, alpha=0.8,
                          edgecolor="black", linewidth=0.5, align="left")
        axes[row][0].set_xlabel(f"# Correct out of {K}")
        axes[row][0].set_ylabel("# Problems")
        axes[row][0].set_title(f"{name} — Per-Problem Consistency")
        axes[row][0].set_xticks(range(K+1))
        axes[row][0].spines[["top", "right"]].set_visible(False)

        labels = [f"Always correct\n({ac})", f"Mixed\n({mix})", f"Always wrong\n({aw})"]
        axes[row][1].pie([ac, mix, aw], labels=labels,
                         colors=[COLORS[1], COLORS[2], COLORS[3]],
                         autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
        axes[row][1].set_title(f"{name} — Categories")

    plt.suptitle(f"Per-Problem Sampling Consistency (K={K})",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_consistency.png", bbox_inches="tight")
    plt.close()
    print("Saved: baseline_consistency.png")


def plot_length_distribution(gsm8k_lengths, math500_lengths, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, lengths, color, name in [
        (axes[0], gsm8k_lengths, COLORS[0], "GSM8K"),
        (axes[1], math500_lengths, COLORS[1], "MATH500"),
    ]:
        ax.hist(lengths, bins=50, color=color, alpha=0.8,
                edgecolor="black", linewidth=0.3)
        ax.axvline(np.median(lengths), color="red", linestyle="--",
                   label=f"Median: {np.median(lengths):.0f}")
        ax.set_xlabel("Completion Length (tokens)")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Qwen-2.5-3B Base — Completion Lengths",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_length_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved: baseline_length_distribution.png")


def plot_math500_categories(math500_raw, math500_results, output_dir):
    """Per-category and per-level breakdowns for MATH500."""
    # Per-subject
    for cat_key in ["subject", "type", "category"]:
        if cat_key not in math500_raw.column_names:
            continue

        cat_correct = defaultdict(int)
        cat_total = defaultdict(int)
        for i, row in enumerate(math500_raw):
            cat = row[cat_key]
            cat_total[cat] += 1
            if math500_results[i]["correct"]:
                cat_correct[cat] += 1

        categories = sorted(cat_total.keys(),
                            key=lambda c: cat_correct[c]/cat_total[c], reverse=True)
        cat_accs = [cat_correct[c]/cat_total[c]*100 for c in categories]
        cat_labels = [f"{c}\n({cat_total[c]})" for c in categories]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(cat_labels, cat_accs, color=COLORS[1], edgecolor="black", linewidth=0.5)
        for bar, acc in zip(ax.patches, cat_accs):
            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                    f"{acc:.1f}%", ha="left", va="center", fontsize=10)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"MATH500 Per-{cat_key.title()} — Qwen-2.5-3B Base", fontweight="bold")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / f"math500_per_{cat_key}.png", bbox_inches="tight")
        plt.close()
        print(f"Saved: math500_per_{cat_key}.png")
        break  # only plot the first found key

    # Per-level
    if "level" in math500_raw.column_names:
        lvl_correct = defaultdict(int)
        lvl_total = defaultdict(int)
        for i, row in enumerate(math500_raw):
            lvl = row["level"]
            lvl_total[lvl] += 1
            if math500_results[i]["correct"]:
                lvl_correct[lvl] += 1

        levels = sorted(lvl_total.keys())
        lvl_accs = [lvl_correct[l]/lvl_total[l]*100 for l in levels]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(levels, lvl_accs, color=COLORS[3], edgecolor="black", linewidth=0.5)
        for bar, acc in zip(bars, lvl_accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xlabel("Difficulty Level")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("MATH500 by Difficulty — Qwen-2.5-3B Base", fontweight="bold")
        ax.set_ylim(0, max(lvl_accs)+15)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / "math500_per_level.png", bbox_inches="tight")
        plt.close()
        print("Saved: math500_per_level.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--pass_k_samples", type=int, default=200,
                        help="Number of problems for pass@K analysis per dataset")
    parser.add_argument("--K", type=int, default=8, help="K for pass@K")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for pass@K generation")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--eval_results_dir", type=str, default="results",
                        help="Dir containing base_gsm8k_detailed.json etc.")
    args = parser.parse_args()

    setup_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    model.eval()

    # Load eval results (from baseline_eval.py)
    eval_dir = Path(args.eval_results_dir)
    gsm8k_detailed = json.loads((eval_dir / "base_gsm8k_detailed.json").read_text())
    math500_detailed = json.loads((eval_dir / "base_math500_detailed.json").read_text())

    # Load raw datasets for category info
    gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_data = [{"question": r["question"], "raw_answer": r["answer"],
                    "dataset": "gsm8k"} for r in gsm8k_raw]

    math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math500_data = [{"question": r["problem"], "raw_answer": r["answer"],
                     "dataset": "math500"} for r in math500_raw]

    # ---- 1. Response length distribution ----
    print("\n--- Response Length Distribution ---")
    gsm8k_lengths = [len(tokenizer.encode(r["completion"], add_special_tokens=False))
                     for r in gsm8k_detailed]
    math500_lengths = [len(tokenizer.encode(r["completion"], add_special_tokens=False))
                       for r in math500_detailed]
    plot_length_distribution(gsm8k_lengths, math500_lengths, output_dir)

    for name, lengths in [("GSM8K", gsm8k_lengths), ("MATH500", math500_lengths)]:
        print(f"  {name}: Mean={np.mean(lengths):.0f} Median={np.median(lengths):.0f} "
              f"Std={np.std(lengths):.0f}")

    # ---- 2. MATH500 per-category ----
    print("\n--- MATH500 Per-Category ---")
    plot_math500_categories(math500_raw, math500_detailed, output_dir)

    # ---- 3. Pass@K — GSM8K ----
    print(f"\n--- Pass@{args.K} GSM8K ({args.pass_k_samples} problems) ---")
    gsm8k_pak, gsm8k_samples, gsm8k_subset, gsm8k_gts = run_pass_at_k(
        model, tokenizer, gsm8k_data, "gsm8k",
        K=args.K, temperature=args.temperature,
        n_samples=args.pass_k_samples, batch_size=args.batch_size,
    )
    for k, v in gsm8k_pak.items():
        print(f"  pass@{k}: {v:.1f}%")
    print(f"  Gap: {gsm8k_pak[args.K] - gsm8k_pak[1]:.1f}%")

    # ---- 4. Pass@K — MATH500 ----
    print(f"\n--- Pass@{args.K} MATH500 ({args.pass_k_samples} problems) ---")
    math500_pak, math500_samples, math500_subset, math500_gts = run_pass_at_k(
        model, tokenizer, math500_data, "math500",
        K=args.K, temperature=args.temperature,
        n_samples=min(args.pass_k_samples, 150), batch_size=args.batch_size,
    )
    for k, v in math500_pak.items():
        print(f"  pass@{k}: {v:.1f}%")
    print(f"  Gap: {math500_pak[args.K] - math500_pak[1]:.1f}%")

    # ---- 5. Pass@K plot ----
    plot_pass_at_k(gsm8k_pak, math500_pak, args.K, args.temperature, output_dir)

    # ---- 6. Consistency ----
    print(f"\n--- Per-Problem Consistency ---")
    gsm8k_cons = compute_consistency(gsm8k_samples, gsm8k_gts, "gsm8k", args.K)
    math500_cons = compute_consistency(math500_samples, math500_gts, "math500", args.K)

    plot_consistency(gsm8k_cons, math500_cons, args.K, output_dir)

    _, gsm8k_ac, gsm8k_mix, gsm8k_aw = gsm8k_cons
    _, math500_ac, math500_mix, math500_aw = math500_cons
    print(f"  GSM8K:   always_correct={gsm8k_ac} mixed={gsm8k_mix} always_wrong={gsm8k_aw}")
    print(f"  MATH500: always_correct={math500_ac} mixed={math500_mix} always_wrong={math500_aw}")

    # ---- 7. Save extended results ----
    extended = {
        "model": args.model,
        "stage": "base",
        "gsm8k_pass_at_k": {str(k): round(v, 2) for k, v in gsm8k_pak.items()},
        "math500_pass_at_k": {str(k): round(v, 2) for k, v in math500_pak.items()},
        "pass_at_k_config": {
            "K": args.K, "temperature": args.temperature,
            "gsm8k_n": len(gsm8k_subset), "math500_n": len(math500_subset),
        },
        "gsm8k_consistency": {"always_correct": gsm8k_ac, "mixed": gsm8k_mix, "always_wrong": gsm8k_aw},
        "math500_consistency": {"always_correct": math500_ac, "mixed": math500_mix, "always_wrong": math500_aw},
        "length_stats": {
            "gsm8k_median": float(np.median(gsm8k_lengths)),
            "gsm8k_mean": float(np.mean(gsm8k_lengths)),
            "math500_median": float(np.median(math500_lengths)),
            "math500_mean": float(np.mean(math500_lengths)),
        },
    }

    with open(output_dir / "base_extended_analysis.json", "w") as f:
        json.dump(extended, f, indent=2)
    print(f"\nSaved: base_extended_analysis.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  GSM8K  pass@1={gsm8k_pak[1]:.1f}% → pass@{args.K}={gsm8k_pak[args.K]:.1f}% "
          f"(gap={gsm8k_pak[args.K]-gsm8k_pak[1]:.1f}%)")
    print(f"  MATH500 pass@1={math500_pak[1]:.1f}% → pass@{args.K}={math500_pak[args.K]:.1f}% "
          f"(gap={math500_pak[args.K]-math500_pak[1]:.1f}%)")
    print(f"\n  Plots saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
