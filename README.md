# Project 2: Generative Modeling Agent

An independent Python project for objective-dependent generative modeling on
PathMNIST and PBMC 3K. Both datasets have verified local copies under `data/`.
Project 1 remains unchanged. All code, dependencies, datasets, checkpoints and
outputs for new runs live here.

Implemented: dataset inspection → AI planning → plan validation → preprocessing →
VAE/GAN/normalizing-flow training → generated samples → objective-specific evaluation and evidence-based recommendations.
Non-pilot training and validation experiments are complete on both datasets, with
consolidated generated reports in `outputs/final_experiment_set_v1`. The separate
four-page submission PDF remains to be prepared. Final-test assessment is optional
for the current submission scope and has not been performed. A one-command workflow
with bounded validation-guided replanning is implemented.

We implemented bounded validation-guided revisions so the agent can use evaluation results to adjust its modeling decisions while controlling computation on the Mac. Round and training-budget limits are configurable; `--max-rounds 1` runs a single modeling round.

## Environment

A separate `.venv` is installed here with Python 3.13.2 and PyTorch 2.14.1.
For a fresh setup (Python 3.11 or newer):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[training,dev,monitoring]'
```

`requirements-macos-tested.txt` records the exact installed dependency versions
for this Mac. These are platform-specific verification records; other platforms
should install the project extras and verify backend compatibility.

## Run the complete validation workflow

```sh
.venv/bin/genmodel-run data/pathmnist.npz --split train --objective "Generate labeled images and evaluate synthetic augmentation" --out outputs/pathmnist_agent_run --total-minutes 30 --max-rounds 2
.venv/bin/genmodel-run data/pbmc3k/filtered_gene_bc_matrices/hg19 --objective "Generate synthetic gene-expression profiles and evaluate statistical fidelity" --out outputs/pbmc_agent_run --total-minutes 30 --max-rounds 2
```

Alternatively use `.venv/bin/python -m genmodel_agent.pipeline` with the same
arguments. The AI can revise its plan after failed validation screens or stop
with an explanation. Python enforces the attempt/budget limits and preserves
the task, split/seed, and evaluation rules. Test data are never used by this
command. Each output directory includes `generated_report.md`, `pipeline.json`,
and independent per-attempt artifacts. See the [workflow guide](docs/pipeline_implementation.md)
for budget scope, stopping rules, and limitations. The stage commands below
remain available for individual inspection and review.

## Inspect

```sh
.venv/bin/python -m genmodel_agent data/pathmnist.npz --split train --objective "Generate labeled images and assess synthetic augmentation" --out outputs/new_image_profile
.venv/bin/python -m genmodel_agent data/pbmc3k/filtered_gene_bc_matrices/hg19 --objective "Generate synthetic cells and evaluate statistical fidelity" --out outputs/new_cell_profile
```

Outputs: `profile.json` and `profile.md`, including shape, feature types, missingness,
zero fraction, class balance, provenance hashes, objective and compute information.
Image axes are recoverable through `Dataset.images`; sparse counts stay sparse.
Additional inspection formats: CSV/TSV (explicit `--orientation`), H5AD (unknown
preprocessing history flagged). Training currently supports MedMNIST and raw 10x MTX.

## Plan with AI

```sh
.venv/bin/python -m genmodel_agent.planning outputs/new_image_profile/profile.json --out outputs/new_image_plan
```

Requires the Codex CLI signed in with ChatGPT, network access and available Codex
allowance. Only compact profile summaries are sent, not image/expression arrays.
Python validates the proposed plan; one repair is allowed. There is no API-key
fallback. Supported tasks: generation, class balancing/augmentation, representation
learning. The planner can select image VAEs, image GANs, count VAEs, and standalone planar,
radial, or RealNVP flows for continuous selected-gene expression generation. GANs support generation/augmentation, not representation learning. See [the planning guide](docs/planner_implementation.md).

## Execute a saved plan

```sh
.venv/bin/python -m genmodel_agent.execution outputs/pathmnist_plan_live/modeling_plan.json --pilot --out outputs/new_image_pilot
.venv/bin/python -m genmodel_agent.execution outputs/pbmc3k_plan_live/modeling_plan.json --pilot --out outputs/new_cell_pilot
```

No AI calls are made during execution. Existing plans are revalidated and source
hashes checked. Use `--dataset PATH` when the same input was relocated.
Use `--device auto` (default), `cpu`, `mps`, or `cuda`. Auto probes actual model
forward/backward operations, with a recorded CPU fallback if acceleration fails.
Explicitly requested devices fail rather than silently changing.

`--pilot` explicitly caps training/validation subsets at 512/128 observations,
epochs and patience at two, generated samples at 32, and each branch at three
minutes. Original and effective plans plus overrides are saved separately.
**Omit `--pilot` to execute the full saved settings.** Choose a new output directory.
The default monitored worker-memory limit is 6 GiB (`--max-memory-gib`).

Each successful model branch saves:

- `best.pt`, `last.pt`: model and optimizer checkpoints with configuration and provenance.
- `history.json`, `learning_curve.png`: model-specific training diagnostics (VAE negative ELBO, GAN losses, or flow NLL).
- `generated.npz`: generated observations, plus labels or gene IDs. VAE runs also save decoder means; GAN runs do not.
- `preprocessing.json`, `splits.npz`: fitted preprocessing and exact split/subset indices.
- `device.json`, `job.json`, `training.log`, `result.json`: device checks, effective job and status.
- `representations.npz` for representation tasks: train/validation posterior means.

The run root contains `execution.json`, `execution_report.md`, source/effective plans,
input profile, package versions and source-code hashes. Test data remain reserved.
See [the execution implementation guide](docs/execution_implementation.md) for details.

## Verification

```sh
.venv/bin/python -m pytest -q
```

Full inspection completed on 89,996 PathMNIST training images and 2,700 PBMC cells
with 32,738 genes. Real CPU and Apple MPS training pilots completed for both
selected models. These two-epoch pilots verify plumbing, not scientific utility.

Current review artifacts:

- [PathMNIST MPS pilot](outputs/pathmnist_training_mps_pilot/execution_report.md)
- [PBMC MPS pilot](outputs/pbmc3k_training_mps_pilot/execution_report.md)
- [PathMNIST representation pilot](outputs/pathmnist_representation_training_pilot/execution_report.md)

Outputs and `.venv` are ignored by Git. Copy required outputs explicitly into any
submission. The assignment's `generated_report_1`, `generated_report_2`, and final
four-page `report.pdf` have not been produced by these milestones.

## Evaluate generated outputs

```sh
.venv/bin/python -m genmodel_agent.evaluation outputs/pathmnist_training_mps_pilot --out outputs/new_image_evaluation
```

Validation is the default. The command computes validity, real-reference fidelity,
diversity, overlap and applicable downstream baseline comparisons, then requests
AI interpretation of those measured results. Add `--no-ai` for offline evaluation.
See [evaluation implementation and final-test safeguards](docs/evaluation_implementation.md).

Completed pilot reports:

- [PathMNIST augmentation](outputs/pathmnist_evaluation_pilot/evaluation_report.md)
- [PBMC generation](outputs/pbmc3k_evaluation_pilot/evaluation_report.md)
- [PathMNIST representation](outputs/pathmnist_representation_evaluation_pilot/evaluation_report.md)

All remain exploratory pilot evidence; no real test set was consumed.

## Non-pilot validation experiments

Fresh AI plans were executed on Apple MPS without pilot caps: PathMNIST completed
30 epochs on 89,996 training images and generated 5,000 labeled samples; PBMC3K
completed 60 epochs on 1,890 training cells and generated 1,000 cells on a
training-selected 1,000-gene panel. Test data remain reserved.

- [Experiment decisions, reproducibility and results](docs/validation_experiments.md)
- [PathMNIST validation report](outputs/pathmnist_evaluation_validation_v1/evaluation_report.md)
- [PBMC validation report](outputs/pbmc3k_evaluation_validation_v1/evaluation_report.md)

Neither experiment supports downstream use yet. PathMNIST synthetic augmentation
reduced validation macro-F1 in the recorded classifier comparison; PBMC failed
several statistical-fidelity screens. The `insufficient_evidence` label also
reflects the deliberate validation-only stage, not a failed execution.

## Image GAN option

A conditional or unconditional convolutional GAN is executable for MedMNIST images.
The planner can select it instead of, or alongside, the image VAE. It uses separate
adversarial updates and validation image statistics for checkpoint selection; final
evaluation uses the same quality and downstream checks as the VAE. See the
[GAN implementation guide](docs/gan_implementation.md).

The [fuller GAN validation experiment](docs/full_gan_experiment.md) completed on
all official training images: early stopping after 21 epochs, best checkpoint at
epoch 13, and 5,000 generated images. Its measured augmentation macro-F1 was
0.6215 versus real-only 0.6337 and the saved VAE's 0.5804. The GAN still failed
quality/utility screens and did not establish augmentation benefit. Final-test
data remain reserved.

## Normalizing-flow options

Three distinct standalone models are now executable: `planar_flow`, `radial_flow`,
and `realnvp_flow` (affine coupling). The AI chooses the method, gene panel,
flow depth, and training settings; Python enforces compatibility and budgets.
These currently support unconditional PBMC generation on 2–128 selected genes.
Their outputs are continuous **log-dequantized expression**, not raw integer
counts. Image flows and flow-based representation learning are not implemented.

```sh
.venv/bin/genmodel-run data/pbmc3k/filtered_gene_bc_matrices/hg19 --objective "Generate continuous log-dequantized gene-expression profiles on a small selected gene panel; choose a suitable normalizing flow and assess statistical fidelity" --out outputs/pbmc_flow_run --total-minutes 30 --max-rounds 2
```

The [implementation and verification guide](docs/normalizing_flows_implementation.md)
explains the transformations, planner choices, output domain, and pilot findings.
All 59 tests passed. Each flow also completed an AI-planned, two-epoch Apple MPS
pilot on PBMC with automated evaluation and reporting. All three pilots failed
marginal-fidelity screens; they verify execution, not downstream usefulness.

## Training diagnostics

Model-fit summaries now accompany sample-quality and downstream-use evidence in
reports and planner feedback. Trend hypotheses guide controlled revisions without
claiming that improved losses guarantee useful samples. Existing thresholds remain
explicitly exploratory. See [implementation details](docs/training_diagnostics.md)
and the [retrospective PBMC preview](outputs/diagnostics_review_v1/review.md).

## Evidence-based evaluation v2

All five evaluation improvements are complete. Task-specific evidence now drives
recommendations instead of a universal all-screens-pass rule. Image evaluation
adds local texture and training-fitted pixel PCA comparisons plus reproducible
class-by-class visual packets. See [policy, methods and new reports](docs/evidence_evaluation_v2.md).

### Inline visuals in future workflow reports

`genmodel-run` now embeds available figures directly in `generated_report.md` for
all attempts and supported datasets/models. It includes real/generated sample
packets when available (with any number of classes), otherwise a saved generated
sample grid, plus learning curves and quality-diagnostic plots. Non-image models
receive their available diagnostic plots without tissue-specific captions.
Missing figures and stages that did not complete are omitted rather than linked
as if they existed. Displaying a sample packet does not change its review status.

The report uses relative image links, so keep its round folders with it when moving
or submitting the report. Standard Markdown viewers display the figures inline.
Rendering uses saved artifacts only; no additional training or evaluation occurs.
To refresh a normal workflow report after moving its entire output directory:

```python
import json
from pathlib import Path
from genmodel_agent.pipeline import render
out = Path("outputs/my_run")
state = json.loads((out / "pipeline.json").read_text())
(out / "generated_report.md").write_text(render(state, out), encoding="utf-8")
```

This reporting update postdates `final_experiment_set_v1`'s frozen source snapshot.
That set's original code hashes and historical artifacts are retained; the new
reporting code applies to future runs without changing its measured results.
