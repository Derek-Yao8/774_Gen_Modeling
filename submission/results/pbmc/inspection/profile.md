# Generative modeling dataset inspection

Inspection only: no model selection, training, generation, or fitted preprocessing has run.

## User objective

Generate continuous log-dequantized gene-expression profiles on a small selected-gene panel from PBMC3K for exploratory statistical analysis. Select compatible generative models and hyperparameters without a prespecified model. Evaluate output validity, model fit, marginal distributions, correlations, diversity and training overlap. Use recorded diagnostics and measured validation evidence for a controlled revision or stopping. Integer count generation is not required. Do not infer biological or downstream usefulness from statistical similarity alone.

Objective interpretation is pending the planning milestone.

## Dataset

- Modality: single_cell_matrix
- Observations: 2,700
- Features: 32,738
- Feature types: {'numeric': 32738}
- Storage: sparse_csr
- Image shape: None
- Inspected split: unsplit input
- Missing values: 0
- Infinite values: 0
- Numeric zero fraction: 0.9741281057464086
- Constant features: 16,104
- Count evidence: 10x input convention and nonnegative integer values
- Labels available: False

## Compute

- System: Darwin (arm64)
- Physical memory: 16.0 GiB
- Available memory snapshot: 4.7 GiB
- PyTorch: 2.14.1
- Suggested device: mps
- Backend probe: successful; individual model operations remain untested

## Split safeguards

Fit preprocessing and models on training data only; use validation for selection and reserve test data for final evaluation.

## Assumptions

- Transposed 10x matrix to observations by features; counts remain unchanged.

## Warnings

- No explicit labels available; supervised class-balancing evaluation requires justified labels.

## Reproducibility

profile.json contains source SHA-256 hashes, package versions, per-feature statistics, label counts, and matrix memory estimates. Source paths and label values are included. No external AI service was called.
