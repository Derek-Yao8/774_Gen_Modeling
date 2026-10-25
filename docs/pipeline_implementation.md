# One-command workflow and bounded feedback

Run from `project2_generative_agent` with its own environment:

```sh
.venv/bin/python -m genmodel_agent.pipeline data/pathmnist.npz --split train --objective "Generate labeled images and evaluate synthetic augmentation" --out outputs/pathmnist_agent_run --device auto --total-minutes 30 --max-rounds 2
```

For PBMC:

```sh
.venv/bin/python -m genmodel_agent.pipeline data/pbmc3k/filtered_gene_bc_matrices/hg19 --objective "Generate synthetic gene-expression profiles and evaluate statistical fidelity" --out outputs/pbmc_agent_run --device auto --total-minutes 30 --max-rounds 2
```

These commands inspect the input, request an AI plan, validate it, train, generate
outputs, evaluate on validation data, and write `generated_report.md`. An installed
package also exposes `genmodel-run`. Existing stage commands remain available.
The final assignment's two report names can be applied when packaging the selected
runs; this development command does not generate the four-page submission PDF.

## Decision loop

We implemented bounded validation-guided revisions so the agent can use evaluation results to adjust its modeling decisions while controlling computation on the Mac. Round and training-budget limits are configurable; `--max-rounds 1` runs a single modeling round.

1. Inspect once and retain the source hashes and objective.
2. Ask the AI planner for a supported model/configuration within this round's allowance.
3. Validate the proposal using the existing scientific, capability and budget checks.
   A malformed proposal receives at most one repair attempt.
4. Execute the validated plan from scratch and evaluate on validation data only.
5. Examine the actual quality and utility screens. The recommendation label alone
   cannot drive this decision: validation-only reports intentionally say
   `insufficient_evidence`, even when individual screens pass.
6. Stop if all applicable screens pass, evaluation is incomplete, training fails,
   the run is a pilot, or the attempt/training allowance is exhausted.
7. Otherwise, provide the AI with the previous plans and numerical validation
   evidence. It can select a changed configuration justified by the measured
   failures or stop with `not_recommended`. It can also request clarification;
   the workflow preserves the question and stops.

Python enforces the maximum attempts (default two, allowed one to three), budget,
unchanged task and full split/seed, unchanged metric set and baseline, and a
configuration different from every previously executed attempt. Changes only to
rationale or time allowance do not qualify. Each revised proposal still passes
all ordinary model/likelihood/preprocessing/hyperparameter checks. The evaluator
uses the same versioned contract; prose cannot relax its acceptance rules.

The feedback includes quality scores, class checks, uncertainty, downstream
comparisons and counts, plus training stop reasons, completed/best epochs and
loss records from the first, best and last five epochs. These convergence
diagnostics help distinguish a short run from a model that stopped improving.
It excludes raw arrays and the prior AI interpretation.
It is marked as untrusted measured data in the planner prompt. All feedback,
prompts, replies and validation errors are saved. The AI's scientific rationale
is auditable but is not mechanically proven correct by these checks.

## Budget and stopping semantics

`--total-minutes` bounds the sum of selected per-model training time ceilings
across attempts. Each round is offered an equal share of the remaining allowance
among remaining rounds (at least one minute). The actual selected ceiling is
charged before training and is not refunded just because the worker finishes
quickly. This prevents repeated cheap runs from evading the attempt budget.

Planning, evaluation, and process startup/termination overhead are outside that
training budget. It is not an overall wall-clock deadline. Each training worker
retains the existing RSS/time/device supervision. `--max-memory-gib` is a monitored
worker limit, not a system reservation. `--pilot` stops after one pilot evaluation
and does not use short pilot failures to launch scientific revisions.

A successful CLI exit means the workflow reached a valid stopping condition,
including a justified rejection or exhausted scientific attempts. It does not
mean synthetic data passed quality checks. Errors, incomplete evaluation,
clarification and invalid plans return a nonzero exit code. The report and JSON
ledger distinguish these outcomes. Failed or interrupted runs preserve available
artifacts; automatic resume is not implemented and repeats need a new directory.

## Saved artifacts

- `inspection/`: input profile and provenance.
- `round_1/`, `round_2/`, optionally `round_3/`: independent planning, execution and
  evaluation folders; checkpoints and generated outputs are never overwritten.
- `pipeline.json`: status, stopping reason, allowances, attempt decisions and evidence.
- `generated_report.md`: overall analysis, metrics, downstream caution and links
  to complete per-attempt decisions, training records and evaluation reports.

There is deliberately no automatic test evaluation or protocol freeze.
Validation/test input splits are also rejected before inspection; MedMNIST runs
must specify `--split train`. Repeated
validation-guided changes can overfit validation data. Review and freeze the final
procedure before using the separate guarded test command. The workflow does not
choose a winning candidate by searching across changing metrics or test outcomes.
Image/count VAEs and image GANs are executable. GANs use the same feedback loop,
with adversarial and validation-feature losses in place of ELBO diagnostics.
Standalone planar, radial and RealNVP expression flows are also implemented; diffusion remains unimplemented.

## Verification

Tests cover an actual CPU training/generation/evaluation workflow on a small image
fixture whose object-valued test arrays cannot be read by the loader. Additional
controlled tests exercise failed-screen revision, AI rejection, unchanged-plan
rejection, split/task-change rejection, budget exhaustion, pilot stopping,
passing-screen stopping, incomplete evaluation and persisted failure reports.
The stage integrations use real Python functions; fake AI responses make the
scientific transition cases reproducible without network access.

The full 45-test suite passed. After adding training diagnostics, all 23 affected
pipeline/planning tests passed; two additional tests passed for rejecting reserved
input splits before inspection. All 79 recorded Project 1 source/document hashes
remain unchanged.

## Live workflow verification

The final live check used the installed `genmodel-run` command on PBMC3K, MPS,
a four-minute total training allowance, two planning rounds, ten epochs per
attempt and at most 500 generated cells. This was a short development workflow
check, not a replacement for the earlier 60-epoch experiment. No executor pilot
caps were applied.

The first plan trained and generated samples, numerical validation and AI
interpretation completed, and failed quality checks triggered feedback. Given
the measured discrepancies and loss history still improving at epoch ten, the AI
returned `not_recommended` for another from-scratch attempt under the unchanged
epoch ceiling. The workflow stopped without creating a second training job and
committed only two of its four available training minutes. Test data were not used.

See the [combined live report](../outputs/pbmc3k_pipeline_validation_v2/generated_report.md),
[AI feedback decision](../outputs/pbmc3k_pipeline_validation_v2/round_2/planning/modeling_plan.md),
and `outputs/pipeline_milestone_verification.json`. The earlier live check is
retained in `pbmc3k_pipeline_validation_v1`; it motivated adding training-loss
diagnostics to the feedback. Revisions are tested with controlled responses;
the live model's decision to stop is not overridden to force another run.

## Diagnostic-aware feedback

New runs include structured training-history summaries and descriptive hypotheses
in planner feedback and reports. See [training diagnostics](training_diagnostics.md)
for evidence separation, limitations, and the preserved screening policy.
