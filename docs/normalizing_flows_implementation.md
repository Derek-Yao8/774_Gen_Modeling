# Standalone normalizing-flow options

The agent now has three separately selectable executable methods: `planar_flow`,
`radial_flow`, and `realnvp_flow`. Each fits a standalone continuous density by
maximum likelihood. These are not VAE posterior-flow extensions. Existing image
VAE, image GAN, and count VAE options remain available.

## Supported problem and output domain

This implementation supports unconditional generation on a small selected-gene
panel from raw 10x single-cell counts, including PBMC3K. It does not implement
image flows, conditional class balancing, imputation, or flow representation
learning. The planner must respect those limits.

Gene selection uses training-only dispersion ranking. For each selected count,
the preprocessing adds deterministic per-cell uniform noise in [0.000001, 1),
takes the natural logarithm, and standardizes using training-only mean and
standard deviation. The seed and original cell index determine the noise,
independently of batch ordering. Validation cells do not fit preprocessing.
The same transformation is used for real-reference evaluation.

Generated arrays undo standardization but remain in log-dequantized expression
coordinates. They may contain negative values. `generated.npz` records this
domain and the selected gene IDs. Samples are not rounded, exponentiated,
library-normalized, or presented as integer counts. Dequantization changes the
modeling target and removes exact zeros; a count-generation objective should
use the count VAE or request clarification. A small panel is not a whole-cell
transcriptome model.

## Transformations and training

All methods map standardized observations to a standard-normal base through a
learned diagonal affine transformation and a stack of flow layers. Training
adds the analytic log-Jacobian determinant to the base log density. Generation
draws base samples and reverses the transformations.

| Method | Transformation | Inverse |
| --- | --- | --- |
| Planar | Rank-one update `x + u tanh(wᵀx + b)`; unit direction and constrained `wᵀu > -0.95` | Monotone scalar equation solved by 64 bisection steps, with residual verification |
| Radial | Expand/contract around a learned center; positive alpha and beta greater than `-0.95 alpha` | Stable quadratic solution for the radius |
| RealNVP | Alternating affine coupling masks; two-hidden-layer ReLU networks predict bounded log-scale and translation | Analytic affine inversion |

RealNVP here is a compact affine-coupling architecture, not a reproduction of
the original image model. Planar/radial formulas are adapted as data-to-base
density transforms. Their foundational reference is
[Rezende and Mohamed (2015)](https://arxiv.org/abs/1505.05770); affine coupling
follows [Dinh, Sohl-Dickstein, and Bengio (2016)](https://arxiv.org/abs/1605.08803).

Adam, gradient clipping, finite-value checks, resource limits, validation-NLL
checkpoint selection, and early stopping apply to all three. Saved training
loss is NLL in standardized continuous coordinates, not a count likelihood or
ELBO. Do not compare it directly with VAE loss or across different gene panels.
Generation checks the composed forward/inverse residual and fails on invalid
outputs. Each run saves checkpoints, history, loss plot, samples, preprocessing,
split indices, device probes, and a result record.

## Agent decisions and integration

The AI planner sees each method's constraints and explains selection and
exclusion. It chooses a 2–128-gene panel, 2–16 layers (`flow_layers`), learning
rate, batch size, epoch/patience budget, and sample count. A flow preserves
dimension, so `latent_dim` must equal `selected_features`. RealNVP's
`hidden_dim` selects the coupling-network width; planar/radial use the existing
schema's fixed value 16 as an explicitly unused placeholder.

Python validates these decisions before dispatch. New plans include catalog
version 2 and the continuous-flow evaluation contract. Archived plans can still
omit the new flow exclusions; the old generic `continuous_flow` option remains
unavailable rather than silently changing its meaning.

The one-command pipeline dispatches the chosen flow, evaluates its outputs,
sends measured evidence to the AI interpreter, and includes its results in
bounded replanning and the final generated report. Validation compares
standardized marginal Wasserstein distance, gene-correlation discrepancy,
diversity, and overlap with real-reference distributions. Integer-count and
zero-fraction checks do not apply. These are heuristic development screens,
not biological, privacy, or downstream-utility guarantees.

The evaluation code signature now includes flow transformations. Older frozen
protocols are not silently rewritten to use the changed evaluator; freeze a
new protocol for a new final assessment. No final-test evaluation was performed
for this milestone.

## Code map

- `genmodel_agent/flows.py`: preprocessing transform, invertible layers, composite density.
- `genmodel_agent/flow_trainer.py`: training, checkpointing, sampling, diagnostics.
- `genmodel_agent/capabilities.py` and `planning.py`: capabilities, AI guidance, validation.
- `models.py`, `trainer.py`, `execution.py`, `pipeline.py`: dispatch and feedback integration.
- `evaluation_data.py`, `evaluation_metrics.py`, `evaluation.py`: domain-aware evaluation and interpretation.
- `tests/test_flows.py`: inverse/Jacobian correctness, gradients, full CPU execution/evaluation, held-out preprocessing isolation, and invalid-plan rejection.

## Verification and measured limits

The complete suite passed: **59 tests**. The eight new flow tests include
analytic log determinants checked against automatic-differentiation Jacobians.
Existing VAE/GAN and pipeline tests also passed.

Three live one-command PBMC pilots completed with AI planning, Apple MPS
training, generation, numerical evaluation, and reporting. Each used 512
training cells, 128 validation cells, two epochs, and 32 generated samples.
Method names were explicitly requested for capability verification; these
runs do not demonstrate an unconstrained comparative model-selection decision.

| Flow | Best validation NLL | Marginal discrepancy/reference ratio | Correlation discrepancy/reference ratio |
| --- | ---: | ---: | ---: |
| Planar | 11.3714 | 1.4290 | 0.9451 |
| Radial | 11.2546 | 1.3728 | 1.0006 |
| RealNVP | 11.2821 | 1.3943 | 1.0012 |

All three failed the marginal-fidelity screen (threshold 1.1), while their
correlation screen passed. All reports recommend `insufficient_evidence`.
Training workers used approximately 0.57–0.63 GiB peak RSS and completed in
about 13–15 seconds including worker overhead. Planning/evaluation overhead
is separate. Fuller training and downstream validation remain future work.

Review the generated reports:

- [Planar pilot](../outputs/pbmc_planar_flow_pilot_v1/generated_report.md)
- [Radial pilot](../outputs/pbmc_radial_flow_pilot_v1/generated_report.md)
- [RealNVP pilot](../outputs/pbmc_realnvp_flow_pilot_v1/generated_report.md)

All 79 Project 1 source/document snapshot hashes still match. Project 1 was
not modified. The verification record is
`outputs/flow_milestone_verification.json`. Test data remain reserved.
