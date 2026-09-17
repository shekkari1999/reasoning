# Results

`reported_run_summary.json` preserves the project owner's historical run
metrics. The underlying checkpoints and raw evaluator outputs were not
preserved in this checkout, so it is explicitly not labeled evaluator output.

For a new run, `scripts/run_eval.sh` writes per-stage summaries and invokes
`src/aggregate_results.py`. The generated `latest_run_summary.json` is then the
canonical evaluator-owned result artifact.
