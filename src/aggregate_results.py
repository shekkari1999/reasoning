"""Build one canonical summary from evaluator-produced stage summaries."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("base", "sft", "dr_grpo")
DATASETS = ("gsm8k", "math500")
REQUIRED_FIELDS = {
    "stage", "model", "prompt_mode", "dtype", "max_new_tokens", "seed",
    "generated_at", "git_commit", "datasets",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_results(results_dir: Path) -> dict:
    stage_results = {}
    evaluation_config = None
    dataset_totals = None

    for stage in STAGES:
        path = results_dir / f"{stage}_eval_summary.json"
        with open(path) as f:
            summary = json.load(f)

        missing = REQUIRED_FIELDS - summary.keys()
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        if summary.get("stage") != stage:
            raise ValueError(f"{path} has stage={summary.get('stage')!r}")
        current_config = {
            "prompt_mode": summary["prompt_mode"],
            "max_new_tokens": summary.get("max_new_tokens"),
            "seed": summary.get("seed"),
            "dtype": summary.get("dtype"),
            "datasets": summary.get("datasets"),
        }
        if evaluation_config is None:
            evaluation_config = current_config
        elif current_config != evaluation_config:
            raise ValueError("All stages must use the same evaluation config")

        current_totals = {
            dataset: summary[dataset]["total"] for dataset in DATASETS
        }
        if dataset_totals is None:
            dataset_totals = current_totals
        elif current_totals != dataset_totals:
            raise ValueError("All stages must evaluate the same sample counts")

        datasets = {}
        for dataset in DATASETS:
            detailed_path = results_dir / f"{stage}_{dataset}_detailed.json"
            with open(detailed_path) as f:
                detailed = json.load(f)
            expected = summary[dataset]
            calculated_accuracy = round(
                expected["correct"] / expected["total"] * 100, 2
            )
            if expected["accuracy"] != calculated_accuracy:
                raise ValueError(f"{path} has inconsistent {dataset} accuracy")
            if len(detailed) != expected["total"]:
                raise ValueError(f"{detailed_path} has the wrong sample count")
            if sum(bool(row["correct"]) for row in detailed) != expected["correct"]:
                raise ValueError(f"{detailed_path} has the wrong correct count")
            datasets[dataset] = {
                "accuracy": expected["accuracy"],
                "correct": expected["correct"],
                "total": expected["total"],
                "detailed_file": detailed_path.name,
                "detailed_sha256": file_sha256(detailed_path),
            }

        stage_results[stage] = {
            "model": summary["model"],
            "model_revision": summary.get("model_revision"),
            "git_commit": summary.get("git_commit"),
            "generated_at": summary.get("generated_at"),
            "summary_file": path.name,
            "summary_sha256": file_sha256(path),
            "datasets": datasets,
        }

    return {
        "provenance": "generated_from_evaluator_summaries",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_config": evaluation_config,
        "stages": stage_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/latest_run_summary.json")
    )
    args = parser.parse_args()

    summary = aggregate_results(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
