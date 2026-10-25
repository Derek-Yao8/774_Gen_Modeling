# generated_report_2 — PBMC3K

These are non-pilot **validation** experiments, with full designated training partitions and bounded evaluation subsets. No final-test scores are reported. Stopping at the configured round, time or patience limit is part of the agent workflow; it is not evidence of model convergence. Quality thresholds are exploratory flags, separate from validity, model fit and downstream usefulness. Confidence intervals condition on fitted runs and do not include training-seed uncertainty.

The objective was continuous log-dequantized selected-gene generation for exploratory statistical analysis. The AI selected a 32-gene RealNVP coupling flow. The second round lowered learning rate from 0.001 to 0.0002 and hidden width from 64 to 32, retaining the same gene panel, splits and preprocessing. These two simultaneous changes do not isolate their individual effects. Each round had a 15-minute training ceiling.

## Validation findings

Training used 1,890 cells; validation used 405 cells; 405 test cells remained reserved. Each model generated 1,000 profiles across the same 32 selected genes. These are continuous log-expression values, not raw integer counts. NLL is measured in standardized continuous coordinates and is comparable here because preprocessing was held fixed.

| Round | Epochs / selected epoch | Validation NLL | Marginal discrepancy/reference ratio | Correlation discrepancy/reference ratio |
|---|---|---:|---:|---:|
| 1 | 9 / 1 | 45.211610 | 3.308 | 1.007 |
| 2 | 10 / 2 | 45.165247 | 3.294 | 1.006 |

The revision slightly improved NLL and marginal discrepancy, but marginal differences remained substantially larger than real-reference variation. Correlation and diversity summaries were closer to their references. These summaries do not establish biological usefulness, and no downstream biological utility task was evaluated. Both outputs passed validity checks.

## Recommendation

Use these outputs only to inspect the statistical generator and its limitations. Keep the selected-gene identities and continuous transformation attached to the samples. Do not substitute them for real cells, interpret them as integer counts, or claim validated biological utility. The agent recommendation remains `insufficient_evidence` and the workflow stops after two rounds.

## Decision and artifact trail

### Round 1: realnvp_flow

Implemented trainer contract 1.0 directly matches unconditional continuous panel generation. Coupling networks can model joint dependencies, while analytic inversion avoids planar numerical inversion. Select one candidate to concentrate the short exploratory budget.

Trained on 1,890 observations with 405 available for validation; completed 9 epochs, selected epoch 1, stopped for `early_stopping`. Generated 1,000 samples.

[Plan](pbmc/round_1/planning/modeling_plan.md) · [Execution](pbmc/round_1/execution/execution_report.md) · [Evaluation and AI interpretation](pbmc/round_1/evaluation/evaluation_report.md) · [Synthetic output](pbmc/round_1/execution/realnvp_flow/generated.npz) · [Metrics](pbmc/round_1/evaluation/realnvp_flow/metrics.json)

#### Visual results

![Round 1 realnvp_flow: training diagnostics](<pbmc/round_1/execution/realnvp_flow/learning_curve.png>)

**Round 1 realnvp_flow: model-fit diagnostics.** NLL is measured in standardized continuous log-expression coordinates; lower values indicate better likelihood fit, not biological usefulness.

![Round 1 realnvp_flow: generated-data quality diagnostics](<pbmc/round_1/evaluation/realnvp_flow/quality_diagnostics.png>)

**Round 1 realnvp_flow: sample-quality diagnostics.** Left: generated-versus-held-out discrepancies compared with real-reference discrepancies (smaller is closer). Right: distances of generated and held-out observations to the training reference subset. These summaries provide neither a privacy guarantee nor proof of downstream utility. The two left-hand metrics have different meanings and should not be compared directly in magnitude. Outputs are continuous selected-gene log-expression, not integer counts.

### Round 2: realnvp_flow

Its implemented version 1.0 contract matches unconditional continuous log-dequantized generation, with analytic inversion and coupling networks for joint dependencies. Revise the existing candidate rather than divide the remaining exploratory budget among models.

Trained on 1,890 observations with 405 available for validation; completed 10 epochs, selected epoch 2, stopped for `early_stopping`. Generated 1,000 samples.

[Plan](pbmc/round_2/planning/modeling_plan.md) · [Execution](pbmc/round_2/execution/execution_report.md) · [Evaluation and AI interpretation](pbmc/round_2/evaluation/evaluation_report.md) · [Synthetic output](pbmc/round_2/execution/realnvp_flow/generated.npz) · [Metrics](pbmc/round_2/evaluation/realnvp_flow/metrics.json)

#### Visual results

![Round 2 realnvp_flow: training diagnostics](<pbmc/round_2/execution/realnvp_flow/learning_curve.png>)

**Round 2 realnvp_flow: model-fit diagnostics.** NLL is measured in standardized continuous log-expression coordinates; lower values indicate better likelihood fit, not biological usefulness.

![Round 2 realnvp_flow: generated-data quality diagnostics](<pbmc/round_2/evaluation/realnvp_flow/quality_diagnostics.png>)

**Round 2 realnvp_flow: sample-quality diagnostics.** Left: generated-versus-held-out discrepancies compared with real-reference discrepancies (smaller is closer). Right: distances of generated and held-out observations to the training reference subset. These summaries provide neither a privacy guarantee nor proof of downstream utility. The two left-hand metrics have different meanings and should not be compared directly in magnitude. Outputs are continuous selected-gene log-expression, not integer counts.

[Complete automatic workflow report](pbmc/generated_report.md) · [Split/preprocessing verification](pbmc/verification.json) · [Experiment verification](verification.json)
