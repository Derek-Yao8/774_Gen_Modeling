# Execution report

Status: **completed**

Execution of the saved modeling plan.

Test data were reserved. Synthetic-data quality and downstream utility have not been evaluated.

## image_vae

- Status: trained
- device: mps
- train_observations: 89996
- validation_observations: 10004
- epochs_completed: 17
- best_epoch: 13
- best_validation_negative_elbo: -2518.2602662060176
- generated_samples: 2500
- stop_reason: early_stopping
- Peak worker RSS: 0.71 GiB
- Wall time: 444.2 seconds

## image_gan

- Status: trained
- device: mps
- train_observations: 89996
- validation_observations: 10004
- epochs_completed: 14
- best_epoch: 13
- best_validation_feature_wasserstein: 0.2544230740322764
- generated_samples: 2500
- stop_reason: wall_time_limit
- Peak worker RSS: 0.80 GiB
- Wall time: 733.2 seconds

## Review artifacts

Each model folder contains the exact job, preprocessing/split records, training log, and result. Successful runs also contain best/last checkpoints, epoch history, a learning curve and generated.npz. Representation objectives additionally save train/validation posterior means. These artifacts do not establish that generated data should be used downstream.
