# Execution report

Status: **completed**

Execution of the saved modeling plan.

Test data were reserved. Synthetic-data quality and downstream utility have not been evaluated.

## realnvp_flow

- Status: trained
- device: mps
- train_observations: 1890
- validation_observations: 405
- epochs_completed: 10
- best_epoch: 2
- best_validation_nll: 45.16524665738329
- generated_samples: 1000
- stop_reason: early_stopping
- Peak worker RSS: 0.62 GiB
- Wall time: 17.0 seconds

## Review artifacts

Each model folder contains the exact job, preprocessing/split records, training log, and result. Successful runs also contain best/last checkpoints, epoch history, a learning curve and generated.npz. Representation objectives additionally save train/validation posterior means. These artifacts do not establish that generated data should be used downstream.
