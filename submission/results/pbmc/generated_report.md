# Automated generative-modeling analysis

Workflow status: **round_limit_reached**

Exploratory screens remained flagged at the attempt limit. Downstream usefulness has not been demonstrated; this is not proof of universal unusability.

Objective: Generate continuous log-dequantized gene-expression profiles on a small selected-gene panel from PBMC3K for exploratory statistical analysis. Select compatible generative models and hyperparameters without a prespecified model. Evaluate output validity, model fit, marginal distributions, correlations, diversity and training overlap. Use recorded diagnostics and measured validation evidence for a controlled revision or stopping. Integer count generation is not required. Do not infer biological or downstream usefulness from statistical similarity alone.

This workflow uses validation only. Final-test data remain reserved. Passing development screens is not approval for downstream use.

Training allowances committed: 30 of 30 minutes. Planning, evaluation and process startup/termination overhead are outside this training budget.

[Dataset profile](inspection/profile.md)

## Attempt 1

Planning: ready

[AI decisions and hyperparameters](round_1/planning/modeling_plan.md)

Propose one RealNVP flow for the explicitly continuous, unconditional generation objective. A 32-gene panel keeps the experiment small under the 15-minute ceiling. Its analytic inverse and coupling networks suit joint-distribution exploration. This is a proposal, not evidence of fit or usefulness.

Training: completed

[Training artifacts](round_1/execution/execution_report.md)

Exploratory validation screens: **failed**

[Measured results and interpretation](round_1/evaluation/evaluation_report.md)

### realnvp_flow

Recommendation: insufficient_evidence

Output validity: True

#### Model-fit diagnostics

History: available; epochs observed: 9.

| Diagnostic | First | Last | Minimum (epoch) | Recent change |
|---|---:|---:|---:|---:|
| train_nll | 45.524 | 41.723 | 41.723 (9) | -2.4261 |
| validation_nll | 45.212 | 46.536 | 45.212 (1) | -0.017088 |

Recent change compares the last two to five available observations; it is not a significance test. Minima of GAN losses are descriptive, not optimal-quality checkpoints.


- NLL is for standardized continuous log-dequantized expression. Compare only identical features and preprocessing; it is not count likelihood or ELBO.
- Recent changes are descriptive, not significance tests. Improved model fit does not establish improved samples or downstream benefit.

#### Sample-quality evidence

These screens are exploratory heuristics; a failed screen is not proof that data are unusable.

- continuous_feature_wasserstein: discrepancy/reference ratio 3.3078540324655004; screen passed: False
- continuous_correlation_mae: discrepancy/reference ratio 1.0073671293519093; screen passed: True

#### Downstream-use evidence

Evaluation status: not_applicable


## Attempt 2

Planning: ready

[AI decisions and hyperparameters](round_2/planning/modeling_plan.md)

Propose one controlled RealNVP revision from scratch: reduce learning_rate from 0.001 to 0.0002 and hidden_dim from 64 to 32. Valid outputs nevertheless failed the marginal Wasserstein screen, while training NLL improved and validation NLL selected epoch 1. Slower updates and lower coupling-network capacity are testable responses to this pattern; improvement is uncertain. Preserve the panel, split, evaluation contract and acceptance criteria.

Training: completed

[Training artifacts](round_2/execution/execution_report.md)

Exploratory validation screens: **failed**

[Measured results and interpretation](round_2/evaluation/evaluation_report.md)

### realnvp_flow

Recommendation: insufficient_evidence

Output validity: True

#### Model-fit diagnostics

History: available; epochs observed: 10.

| Diagnostic | First | Last | Minimum (epoch) | Recent change |
|---|---:|---:|---:|---:|
| train_nll | 45.434 | 44.988 | 44.988 (10) | -0.2361 |
| validation_nll | 45.175 | 45.316 | 45.165 (2) | 0.1006 |

Recent change compares the last two to five available observations; it is not a significance test. Minima of GAN losses are descriptive, not optimal-quality checkpoints.

- **possible_overfitting**: Training objective decreased while validation objective increased over the recent window. This is compatible with overfitting, not proof of its cause. Consider a controlled capacity or optimization change; do not simply extend identical training.

- NLL is for standardized continuous log-dequantized expression. Compare only identical features and preprocessing; it is not count likelihood or ELBO.
- Recent changes are descriptive, not significance tests. Improved model fit does not establish improved samples or downstream benefit.

#### Sample-quality evidence

These screens are exploratory heuristics; a failed screen is not proof that data are unusable.

- continuous_feature_wasserstein: discrepancy/reference ratio 3.2943816025225123; screen passed: False
- continuous_correlation_mae: discrepancy/reference ratio 1.0056836087850203; screen passed: True

#### Downstream-use evidence

Evaluation status: not_applicable


## Downstream recommendation

Do not adopt generated outputs on the basis of this development run alone. Exploratory concerns can warrant model revision or stopping; incomplete evaluation cannot establish usefulness. Any final assessment requires a frozen protocol and the separate guarded test-evaluation command.

See pipeline.json for the complete attempt ledger, budgets and stopping reason. Each attempt retains its own plans, checkpoints, generated data and evaluation records.
