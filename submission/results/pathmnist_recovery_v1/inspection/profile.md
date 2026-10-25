# Generative modeling dataset inspection

Inspection only: no model selection, training, generation, or fitted preprocessing has run.

## User objective

Generate labeled PathMNIST images and determine whether adding synthetic images improves classification relative to real-only training and class-matched real resampling. Select appropriate compatible generative models and hyperparameters from the available catalog without a prespecified model. Evaluate output validity, model fit, overall and per-class image quality, diversity, training overlap, and downstream augmentation utility. Use measured validation evidence for a controlled revision or stopping. Report limitations and do not infer clinical validity from appearance or classifier performance.

Objective interpretation is pending the planning milestone.

## Dataset

- Modality: image
- Observations: 89,996
- Features: 2,352
- Feature types: {'numeric': 2352}
- Storage: dense
- Image shape: [28, 28, 3]
- Inspected split: train
- Missing values: 0
- Infinite values: 0
- Numeric zero fraction: 0.00014701144691842691
- Constant features: 0
- Count evidence: not established
- Labels available: True

### Label: label_0

Missing: 0; distinct observed values: 9; majority/minority ratio: 1.6339081917321836

- 8: 12,885
- 5: 12,182
- 3: 10,401
- 2: 10,360
- 1: 9,509
- 7: 9,401
- 0: 9,366
- 4: 8,006
- 6: 7,886

## Compute

- System: Darwin (arm64)
- Physical memory: 16.0 GiB
- Available memory snapshot: 4.3 GiB
- PyTorch: 2.14.1
- Suggested device: mps
- Backend probe: successful; individual model operations remain untested

## Split safeguards

Fit preprocessing and models on training data only; use validation for selection and reserve test data for final evaluation.

## Assumptions

- Flattened image axes in C order; pixel values are not scaled and spatial layout is recorded.

## Warnings


## Reproducibility

profile.json contains source SHA-256 hashes, package versions, per-feature statistics, label counts, and matrix memory estimates. Source paths and label values are included. No external AI service was called.
