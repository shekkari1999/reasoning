# Results

`latest_run_summary.json` contains the structured results used by the root
README and resume.

For a new run, `scripts/run_eval.sh` writes per-stage summaries and invokes
`src/aggregate_results.py` to create `evaluator_run_summary.json`, including
validated counts and hashes for the detailed prediction files.
