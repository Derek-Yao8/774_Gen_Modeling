# Execution report

Status: **completed**

Execution of the saved modeling plan.

Test data were reserved. Synthetic-data quality and downstream utility have not been evaluated.

## image_gan

- Status: trained
- device: mps
- train_observations: 89996
- validation_observations: 10004
- epochs_completed: 6
- best_epoch: 2
- best_validation_feature_wasserstein: 0.3890950488083704
- generated_samples: 2500
- stop_reason: early_stopping
- Peak worker RSS: 0.78 GiB
- Wall time: 307.3 seconds

## Review artifacts

Each model folder contains the exact job, preprocessing/split records, training log, and result. Successful runs also contain best/last checkpoints, epoch history, a learning curve and generated.npz. Representation objectives additionally save train/validation posterior means. These artifacts do not establish that generated data should be used downstream.
