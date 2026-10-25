# Project 2 submission materials

This repository contains the source and generated deliverables for **Building an AI Agent for Generative Modeling**. The author will add the separate four-page `report.pdf` last; it is intentionally absent from this package.

## Deliverables

| Required material | Location |
|---|---|
| Agent source | `../genmodel_agent/` |
| Dependencies and environment instructions | `../pyproject.toml`, `../requirements-macos-tested.txt`, `../README.md` |
| Tests | `../tests/` |
| PathMNIST report, including visualizations | [generated_report_1](results/generated_report_1.md) |
| PBMC3K report, including visualizations | [generated_report_2](results/generated_report_2.md) |
| Synthetic outputs, evaluations, decisions and histories | [Results guide](results/README.md) |
| Source used during experiments | `results/source_snapshot.zip` |
| Package integrity record | `manifest.json` |
| Fresh-run objectives and budgets | `run_config.json` |

The current source includes the general illustrated-report update added after the experiments. Saved numerical results are unchanged. Markdown reports render on GitHub or in a Markdown viewer. Their figures are bundled and use relative paths. Keep `results/` intact; copying just a report file will omit its images.

## Reproduce from a clean checkout

From the repository root, use Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[training,dev,monitoring]'
.venv/bin/python scripts/setup_datasets.py
```

Install Codex CLI if not already available, run `codex login`, and sign in with ChatGPT. Confirm authentication with `codex login status`. The planner requires service access and available account usage; it does not use an API-key fallback.

Then run both workflows with a single command:

```sh
.venv/bin/python scripts/run_submission.py --out outputs/submission_rerun
```

The wrapper runs the normal agent sequentially with the submitted objectives and ceilings, using local `data/` paths and automatic device selection. Use `--device mps` on a compatible Mac to match the recorded backend, or `--device cpu` on another machine. A new output directory is required. Each dataset receives its own generated report, figures, synthetic data, model decisions and evaluation records. The four-page manual report is prepared separately.

For a command-only check without planner calls or training:

```sh
.venv/bin/python scripts/setup_datasets.py --verify-only
.venv/bin/python scripts/run_submission.py --dry-run
```

The language-model planner can make different valid choices on a new run. Seeds, saved plans, source and package versions document the original results; hardware/library differences can also affect numerical reproduction. Raw data are downloaded during setup rather than stored in Git. Checkpoints, caches, verbose planner event streams, interrupted duplicate training outputs and `report.pdf` are excluded. The included generated samples and metrics can be inspected without retraining; checkpoint-dependent reevaluation requires regenerating checkpoints.

## Interpretation and history

These are completed validation experiments, not final-test results. The generated data have not demonstrated downstream usefulness. PathMNIST was recovered after a process interruption and continued after a planner usage limit; its consolidated report identifies the authoritative rounds. Historical paths and failure statuses are retained in provenance records, not used by the fresh-run wrapper. The visual-review appendix records a separate non-expert assistant inspection.

For Canvas, submit a ZIP of the repository contents (without `.git`, `.venv`, downloaded data or local caches) and add your final `report.pdf` at the root. GitHub's **Code > Download ZIP** provides the tracked files after they have been pushed. Do not submit only this `submission/` subfolder: the source, dependencies, tests and setup scripts live at the repository root.
