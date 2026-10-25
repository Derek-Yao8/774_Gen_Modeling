# First non-pilot validation experiments

These experiments execute fresh AI plans using the current trainer and evaluation
contracts. They are development experiments, not final-test results. They do not
complete the assignment's final submission.

## Planning and implementation

Both objectives are unchanged from the pilots: PathMNIST labeled generation and
augmentation, and PBMC3K count generation with statistical-fidelity evaluation.
Profiles were refreshed in the Project 2 environment so the planner sees the
installed PyTorch and available Apple MPS backend.

The first refreshed PathMNIST plan requested clarification about pixel dtype and
range. These were measurable input properties omitted from the compact profile.
The profiler now records `value_semantics.image_dtype` and `pixel_range`; these
automatically reach the planner. An inspection regression test checks both the
measurements and their presence in the compact planner input. No user-supplied
scientific facts were invented to resolve the clarification.

Planning attempt history is retained under `outputs/`: the initial sandbox-blocked
`pathmnist_plan_validation_v1`, the access-check plan using the older profile,
and the clarification in `pathmnist_plan_validation_v2`. None was used for training.
The executed PathMNIST plan is `pathmnist_plan_validation_v3`; the PBMC plan is
`pbmc3k_plan_validation_v1`.

## Resource and scientific boundaries

- Each planner has a 30-minute total training ceiling and at most 5,000 generated
  samples. Epoch ceilings supplied to planning are 40 for images and 60 for cells.
- The planner selected a conditional image VAE (30 epochs, patience 5, latent 32,
  hidden 128, batch 128, 5,000 images) and an unconditional count VAE (60 epochs,
  patience 8, latent 16, hidden 128, batch 64, 1,000 genes, 1,000 generated cells).
  Both use learning rate 0.001. Models and settings were AI-selected, not manually
  substituted after observing validation results.
- Training runs sequentially on MPS, with a monitored 6 GiB worker memory ceiling.
  The trainer reserves 20% of its time allowance for finishing artifacts.
- No pilot caps are applied. PathMNIST uses the official training/validation
  partitions. PBMC uses seed 42 and a 70/15/15 train/validation/test split, with
  gene selection fitted only on training cells.
- Evaluation still uses recorded computational caps: at most 4,096 training
  reference rows, 1,024 validation rows, 2,048 generated rows, and 512 rows for
  equal-size quality comparisons. A full training run does not imply evaluation
  on every sample. Classifier comparisons use the recorded training subset.
- Both are validation-only runs. The recommendation label remains
  `insufficient_evidence` while final testing is pending, independently of which
  individual quality screens pass. Failed screens remain visible in the report.
- The uncertainty intervals condition on one fitted model, not repeated training.
  Quality discrepancy bootstraps can be biased and need not contain the original
  point estimate. Do not treat those intervals as calibrated guarantees or as
  confidence intervals for reference-normalized ratios.

## Reproducing the stages

Run from the Project 2 directory using its `.venv`. Existing output directories
are never overwritten: choose new names for a repeat. AI proposals can vary even
with identical inputs; saved plans preserve the choices actually executed.

```sh
.venv/bin/python -m genmodel_agent data/pathmnist.npz --split train --objective "Generate labeled images and evaluate synthetic augmentation on held-out real images." --out outputs/repeat_image_profile
.venv/bin/python -m genmodel_agent.planning outputs/repeat_image_profile/profile.json --out outputs/repeat_image_plan --total-minutes 30 --max-epochs 40 --max-samples 5000
.venv/bin/python -m genmodel_agent.execution outputs/repeat_image_plan/modeling_plan.json --device mps --max-memory-gib 6 --out outputs/repeat_image_training
.venv/bin/python -m genmodel_agent.evaluation outputs/repeat_image_training --out outputs/repeat_image_evaluation

.venv/bin/python -m genmodel_agent data/pbmc3k/filtered_gene_bc_matrices/hg19 --objective "Generate synthetic gene-expression profiles and evaluate statistical fidelity." --out outputs/repeat_cell_profile
.venv/bin/python -m genmodel_agent.planning outputs/repeat_cell_profile/profile.json --out outputs/repeat_cell_plan --total-minutes 30 --max-epochs 60 --max-samples 5000
.venv/bin/python -m genmodel_agent.execution outputs/repeat_cell_plan/modeling_plan.json --device mps --max-memory-gib 6 --out outputs/repeat_cell_training
.venv/bin/python -m genmodel_agent.evaluation outputs/repeat_cell_training --out outputs/repeat_cell_evaluation
```

Planning and AI interpretation require the signed-in Codex connection. Numerical
evaluation can instead run with `--no-ai`. The Mac sandbox may require approval
for local Codex service access and MPS. Project 1 data are read-only inputs;
all environments, code changes and new artifacts belong to Project 2.

## PBMC results

The count VAE completed 60 epochs on 1,890 training cells and 405 validation
cells; epoch 60 had the best validation negative ELBO (267.1932). The supervised
worker took 38.7 seconds with peak process RSS 0.61 GiB on this run. RSS is not
total unified/GPU memory. Generated output is 1,000 cells by 1,000 selected genes.

Structural validity passed. Evaluation used all 1,890 training-reference cells,
405 validation cells, and 1,000 generated cells, with equal-size quality
comparisons on 405 rows. The discrepancy-to-real-reference ratios were:

| Metric | Ratio | Heuristic screen |
|---|---:|---|
| Log gene means | 1.332 | Fail |
| Log gene variances | 0.482 | Pass |
| Gene zero fractions | 1.915 | Fail |
| Log selected-panel totals | 1.953 | Fail |
| Gene correlations | 2.301 | Fail |

Diversity ratio was 1.296 (outside 0.9–1.1). No exact training-reference
duplicates were detected. These results do not support downstream substitution
of synthetic cells for real data; gene correlations and sparsity remain concerns.
No supervised downstream utility was evaluated for this unlabeled generation
objective. The numerical label is `insufficient_evidence` because final testing
is pending; it does not mean the failed validation screens were ignored.

See the [PBMC plan](../outputs/pbmc3k_plan_validation_v1/modeling_plan.md),
[training record](../outputs/pbmc3k_training_validation_v1/execution_report.md),
and [evaluation report](../outputs/pbmc3k_evaluation_validation_v1/evaluation_report.md).

## PathMNIST results

The conditional image VAE completed 30 epochs on 89,996 training images and
10,004 validation images. Epoch 26 was selected, with validation negative ELBO
-2523.6186. The supervised worker took 509.6 seconds with peak process RSS
0.92 GiB on this run; this is not total unified/GPU memory. It generated 5,000
labeled uint8 RGB images. Decoder means are retained separately from observation
draws; the noisy observation draws, not the smoother means, were evaluated.

Evaluation used 4,096 real training-reference images, 1,024 real validation
images, and 2,048 generated images, with 512-row equal-size quality comparisons.
Structural validity passed. Pooled-feature Wasserstein discrepancy was 4.441
times the real-reference median; diversity ratio was 0.725. Aggregate quality
and all nine per-class combined screens failed. No exact training-reference
duplicates were detected, which provides no privacy guarantee.

The fixed logistic classifiers all converged. Their validation macro-F1 scores:

| Training arm | Macro-F1 |
|---|---:|
| Real only | 0.6337 |
| Real plus synthetic | 0.5804 |
| Real plus class-matched real resampling | 0.6514 |

Synthetic augmentation minus real-only was -0.05325, with paired 95% bootstrap
interval [-0.06970, -0.03126]. Synthetic augmentation minus the resampling control
was -0.07094, interval [-0.09129, -0.04631]. These support worse performance for
this particular fitted augmentation procedure on the recorded validation
sample; they do not establish that every VAE or every downstream classifier
would behave the same way. The intervals exclude variation from retraining.
The resampling arm also failed the strict per-class utility gate, so its higher
macro-F1 alone does not constitute acceptance of that alternative.

See the [PathMNIST plan](../outputs/pathmnist_plan_validation_v3/modeling_plan.md),
[training record](../outputs/pathmnist_training_validation_v1/execution_report.md),
and [evaluation report](../outputs/pathmnist_evaluation_validation_v1/evaluation_report.md).

## Interpretation and remaining work

These are completed, bounded validation experiments with negative or mixed
findings. More epochs alone did not establish downstream usefulness. Do not
claim a controlled improvement over the pilots: training sizes, evaluation
sizes and sometimes hyperparameters differ. Preserve the current results as
evidence instead of changing acceptance thresholds after seeing failures.

A further experiment should address a specific limitation, such as the image
observation model/capacity or the count model's correlation structure, or add
another executable family for a meaningful alternative. Such changes should
receive a new plan and versioned contracts, then be evaluated on validation
data before final testing. This milestone does not implement an autonomous
evaluation-to-replanning loop or another model family. The required one-command
orchestration and final submission reports also remain outstanding.

## Verification

The pre-experiment test suite passed all 33 tests. After the pixel-evidence fix,
all 14 affected inspection/planning tests passed. Both execution records and
evaluation records are complete, non-pilot, and mark test data unused; both AI
interpretations completed. Saved checkpoints, generated arrays and split records
were checked. All 79 recorded Project 1 source/document hashes match the
pre-experiment snapshot. The machine-readable check is
`outputs/validation_experiments_verification.json`.
