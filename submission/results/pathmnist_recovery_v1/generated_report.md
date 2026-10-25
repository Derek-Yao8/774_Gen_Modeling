# Automated generative-modeling analysis

Workflow status: **planner_unavailable**

Codex failed (exit 1). See pathmnist_recovery_v1/round_2/planning/attempt_1/stderr.txt. No paid fallback.

Objective: Generate labeled PathMNIST images and determine whether adding synthetic images improves classification relative to real-only training and class-matched real resampling. Select appropriate compatible generative models and hyperparameters from the available catalog without a prespecified model. Evaluate output validity, model fit, overall and per-class image quality, diversity, training overlap, and downstream augmentation utility. Use measured validation evidence for a controlled revision or stopping. Report limitations and do not infer clinical validity from appearance or classifier performance.

This workflow uses validation only. Final-test data remain reserved. Passing development screens is not approval for downstream use.

Training allowances committed: 30 of 60 minutes. Planning, evaluation and process startup/termination overhead are outside this training budget.

[Dataset profile](inspection/profile.md)

## Attempt 1

Planning: ready

[AI decisions and hyperparameters](round_1/planning/modeling_plan.md)

Propose conditional image VAE and GAN candidates for a controlled augmentation comparison. The VAE offers a stable likelihood-based baseline; the GAN tests a complementary sample-fidelity tradeoff. The 30-minute ceiling supports exploratory evidence, not assured convergence or downstream usefulness.

Training: completed

[Training artifacts](round_1/execution/execution_report.md)

Exploratory validation screens: **failed**

[Measured results and interpretation](round_1/evaluation/evaluation_report.md)

### image_vae

Recommendation: insufficient_evidence

Output validity: True

#### Model-fit diagnostics

History: available; epochs observed: 17.

| Diagnostic | First | Last | Minimum (epoch) | Recent change |
|---|---:|---:|---:|---:|
| train_negative_elbo | -2035.7 | -2539.4 | -2539.4 (17) | -12.94 |
| validation_negative_elbo | -2277.4 | -2516.7 | -2518.3 (13) | 1.5565 |
| train_reconstruction_nll | -2064.2 | -2573.9 | -2573.9 (17) | -13.741 |
| validation_reconstruction_nll | -2307.4 | -2551.4 | -2551.7 (13) | 0.35773 |
| train_kl | 28.559 | 34.544 | 28.559 (1) | 0.80178 |
| validation_kl | 29.956 | 34.687 | 29.63 (4) | 1.1988 |

Recent change compares the last two to five available observations; it is not a significance test. Minima of GAN losses are descriptive, not optimal-quality checkpoints.

- **possible_overfitting**: Training objective decreased while validation objective increased over the recent window. This is compatible with overfitting, not proof of its cause. Consider a controlled capacity or optimization change; do not simply extend identical training.

- ELBO, reconstruction NLL and KL describe model fit. Low KL can motivate posterior-collapse investigation but no universal KL cutoff is used.
- Recent changes are descriptive, not significance tests. Improved model fit does not establish improved samples or downstream benefit.

#### Sample-quality evidence

These screens are exploratory heuristics; a failed screen is not proof that data are unusable.

- feature_wasserstein: discrepancy/reference ratio 4.004554048991598; screen passed: False

#### Downstream-use evidence

Evaluation status: computed

- Macro-F1 difference: -0.047373118294390615; paired interval: [-0.06814145855922348, -0.025146126711955697]

### image_gan

Recommendation: insufficient_evidence

Output validity: True

#### Model-fit diagnostics

History: available; epochs observed: 14.

| Diagnostic | First | Last | Minimum (epoch) | Recent change |
|---|---:|---:|---:|---:|
| generator_loss | 2.5889 | 1.3935 | 1.1878 (9) | 0.17669 |
| discriminator_loss | 0.33021 | 0.51298 | 0.33021 (1) | -0.049288 |
| validation_feature_wasserstein | 0.43543 | 0.27895 | 0.25442 (13) | -0.013347 |

Recent change compares the last two to five available observations; it is not a significance test. Minima of GAN losses are descriptive, not optimal-quality checkpoints.


- Adversarial losses describe optimization, not sample quality. Their relative magnitudes do not diagnose discriminator dominance or mode collapse.
- Recent changes are descriptive, not significance tests. Improved model fit does not establish improved samples or downstream benefit.

#### Sample-quality evidence

These screens are exploratory heuristics; a failed screen is not proof that data are unusable.

- feature_wasserstein: discrepancy/reference ratio 2.7224509166562267; screen passed: False

#### Downstream-use evidence

Evaluation status: computed

- Macro-F1 difference: -0.02085167676241939; paired interval: [-0.047451908778465925, 0.004381589950247106]

## Attempt 2

Planning: planner_unavailable

[AI decisions and hyperparameters](round_2/planning/modeling_plan.md)

Codex failed (exit 1). See pathmnist_recovery_v1/round_2/planning/attempt_1/stderr.txt. No paid fallback.

## Downstream recommendation

Do not adopt generated outputs on the basis of this development run alone. Exploratory concerns can warrant model revision or stopping; incomplete evaluation cannot establish usefulness. Any final assessment requires a frozen protocol and the separate guarded test-evaluation command.

See pipeline.json for the complete attempt ledger, budgets and stopping reason. Each attempt retains its own plans, checkpoints, generated data and evaluation records.
