#!/usr/bin/env python3
"""Plot training curves from MetricTracker JSON files.

Usage:
    python scripts/plot_training.py checkpoints/sft/sft_metrics.json
    python scripts/plot_training.py checkpoints/dr_grpo/dr_grpo_metrics.json -o results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_series(data: dict, key: str) -> tuple[list[int], list[float]]:
    entries = data.get(key, [])
    steps = [e["step"] for e in entries]
    values = [e["value"] for e in entries]
    return steps, values


def plot_metrics(metrics_path: Path, output_dir: Path) -> None:
    with open(metrics_path) as f:
        data = json.load(f)

    stem = metrics_path.stem.replace("_metrics", "")
    keys = list(data.keys())
    if not keys:
        print(f"No metrics in {metrics_path}")
        return

    n = len(keys)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(stem, fontsize=14, fontweight="bold")

    for i, key in enumerate(keys):
        ax = axes[i // cols][i % cols]
        steps, values = load_series(data, key)
        ax.plot(steps, values, linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{stem}_training_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics JSON")
    parser.add_argument("metrics_json", type=str, help="Path to *_metrics.json")
    parser.add_argument("-o", "--output-dir", type=str, default="results")
    args = parser.parse_args()

    plot_metrics(Path(args.metrics_json), Path(args.output_dir))


if __name__ == "__main__":
    main()
