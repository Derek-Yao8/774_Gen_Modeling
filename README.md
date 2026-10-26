# Project 2: Generative Modeling Agent

A Python agent that inspects a dataset, interprets a user objective, selects
compatible generative models and hyperparameters, trains, generates outputs,
evaluates evidence and recommends downstream use. Bounded validation-guided
revisions let it adjust modeling decisions while controlling computation.

Implemented models include image and count VAEs, a non-saturating BCE image GAN,
and planar, radial and RealNVP normalizing flows. Support is limited to the
implemented data formats and trainer contracts; not every model supports every
data type or objective. See [planner design](docs/planner_implementation.md),
[training implementation](docs/execution_implementation.md),
[flow implementation](docs/normalizing_flows_implementation.md) and
[evaluation policy](docs/evidence_evaluation_v2.md).

## Submitted deliverables

The repository as a whole is the submission package. The author will add the
separate four-page **report.pdf** at the repository root last; it is intentionally
not included yet.

| Material | Location |
|---|---|
| Agent source | [genmodel_agent/](genmodel_agent/) |
| Tests | [tests/](tests/) |
| Dependency ranges | [pyproject.toml](pyproject.toml) |
| Tested Mac dependency snapshot | [requirements-macos-tested.txt](requirements-macos-tested.txt) |
| PathMNIST illustrated report | [generated_report_1](submission/results/generated_report_1.md) |
| PBMC3K illustrated report | [generated_report_2](submission/results/generated_report_2.md) |
| Synthetic outputs, evaluations and decision records | [submission/results/](submission/results/) |
| Modeling source used for the experiments | [source_snapshot.zip](submission/results/source_snapshot.zip) |
| Portable-package checksums | [manifest.json](submission/manifest.json) |
| Objectives and budgets for fresh runs | [run_config.json](submission/run_config.json) |

Reports display figures through relative links. They render directly on GitHub
or in a Markdown preview; keep the results folders intact when moving them.
Reading the submitted reports needs no planner login or model training.

## Install and prepare datasets

From the repository root, use Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[training,dev,monitoring]'
.venv/bin/python scripts/setup_datasets.py
```

The setup script downloads the official PathMNIST and PBMC3K datasets and verifies
all four files against [recorded checksums](data/checksums.json). It preserves
publisher splits, extracts only the required 10x files and refuses to overwrite
mismatching data. New runs use `data/pathmnist.npz` and
`data/pbmc3k/filtered_gene_bc_matrices/hg19/` inside Project 2. Raw datasets are
not stored in Git. Project 1 remains unchanged and is not needed for fresh runs.

Install Codex CLI if not already available, run `codex login`, and sign in with
ChatGPT. Confirm with `codex login status`. Fresh AI decisions require network
access and available Codex usage. The adapter has no API-key fallback and does
not pin an underlying LLM model. Planner authentication and dataset download are
setup steps, not repeated manual stages within each modeling run.

The experiments used Python 3.13.2, PyTorch 2.14.1 and MPS on an Apple M1 Mac.
The dependency snapshot records the tested Mac environment; other platforms
should install the project extras above and verify backend compatibility.

## Run the complete workflow

For one dataset, provide its path and the intended modeling objective:

```sh
.venv/bin/genmodel-run data/pathmnist.npz --split train --objective "Generate labeled images and evaluate synthetic augmentation" --out outputs/new_pathmnist_run --total-minutes 60 --max-rounds 2
```

For continuous selected-gene expression generation:

```sh
.venv/bin/genmodel-run data/pbmc3k/filtered_gene_bc_matrices/hg19 --objective "Generate continuous log-dequantized expression profiles on a small selected-gene panel and evaluate statistical fidelity" --out outputs/new_pbmc_run --total-minutes 30 --max-rounds 2
```

Alternatively, run **both datasets with one command**, using the submitted
objectives and training ceilings:

```sh
.venv/bin/python scripts/run_submission.py --out outputs/submission_rerun
```

The wrapper runs the normal agent sequentially using local data paths and
automatic device selection. Add `--device mps` on a compatible Mac or
`--device cpu` on another machine. Always select a new output directory; existing
runs are never overwritten. `--max-rounds 1` on `genmodel-run` selects one round.

Each workflow performs inspection, planning, execution, generation, validation,
bounded revision or stopping, and reporting. Its `generated_report.md` embeds
available real/generated image packets, learning curves and quality plots.
Non-image models receive their available diagnostic plots. Missing figures are
omitted, and displaying images does not change their visual-review status.
Plans, checkpoints, numerical outputs and evaluations are retained per attempt.
The separate manual PDF is prepared afterward. See the
[workflow guide](docs/pipeline_implementation.md) for stage commands and budgets.

Check setup and commands without training or planner calls:

```sh
.venv/bin/python scripts/setup_datasets.py --verify-only
.venv/bin/python scripts/run_submission.py --dry-run
.venv/bin/python -m pytest
```

A fresh AI plan may choose different valid settings. Recorded seeds, plans and
package versions support reproduction but do not guarantee identical choices or
numerical results across hardware, libraries or planner versions.

## Understanding the saved results

These are completed **validation** experiments, not final-test results. Neither
dataset established downstream usefulness of the generated data under the
measured conditions. The reports retain negative and inconclusive findings.

Paths below are relative to `submission/results/`:

| Dataset / round | Authoritative artifacts |
|---|---|
| PathMNIST round 1: conditional VAE and GAN | `pathmnist_recovery_v1/round_1/` |
| PathMNIST round 2: revised GAN | `pathmnist_continuation_v1/` |
| PBMC round 1: RealNVP | `pbmc/round_1/` |
| PBMC round 2: revised RealNVP | `pbmc/round_2/` |

PathMNIST's initial process was interrupted; recovery replayed its first plan.
The recovered first round completed, but a planner usage limit prevented the
second planning call. After the limit reset, continuation reused that evidence
and completed round 2. The folders preserve this history. Earlier failure statuses
are historical records, not the status of the completed experiment set. The
PathMNIST report explains the details and additional interrupted-run overhead.

Each model's `execution/<model>/generated.npz` contains synthetic outputs.
Effective plans, hyperparameters, training histories, split indices, evaluation
arrays, metrics, recommendations and AI interpretations are included. Model
checkpoints and verbose CLI event streams are omitted from Git; historical
execution reports describe the original checkpoint files retained in the local
archive. Regenerate checkpoints by training if a fresh checkpoint-dependent
evaluation is needed. Inspecting the supplied samples and metrics needs no rerun.

Historical `experiment_config.json`, execution jobs and provenance retain their
original source paths. Do not use those as fresh-run configurations. The current
`submission/run_config.json` and `scripts/run_submission.py` use Project 2's local
data paths and new output directories. `source_snapshot.zip` preserves the
modeling code used in the experiments. Current root source additionally includes
the subsequent general illustrated-report update; measured results are unchanged.

[verification.json](submission/results/verification.json) records the original
experiment checks. [manifest.json](submission/manifest.json) lists portable-package
checksums after report image links were rewritten. The separate
[visual review](submission/results/visual_review.md) records post-run, non-expert
assistant observations, not expert approval or a review by the automated interpreter.

## Refresh a report without retraining

For a normal workflow directory, render its saved state and existing figures:

```python
import json
from pathlib import Path
from genmodel_agent.pipeline import render
out = Path("outputs/my_run")
state = json.loads((out / "pipeline.json").read_text())
(out / "generated_report.md").write_text(render(state, out), encoding="utf-8")
```

`scripts/package_submission.py` rebuilds the portable result package from the
local `outputs/final_experiment_set_v1` archive. That development archive is not
required to read this checkout or run new experiments, and is not included in Git.
Earlier milestone documents may link to omitted local `outputs/` archives; this
root README is the current submission and usage guide.
