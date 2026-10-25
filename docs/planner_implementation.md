# How the Project 2 planner works

## Review map

- `genmodel_agent/planning.py`: catalog, prompt, strict field checks, scientific validation, bounded repair, CLI, report rendering.
- `genmodel_agent/planner_backend.py`: independent Codex CLI adapter copied and adapted from Project 1, without runtime imports from that project.
- `tests/test_planning.py`: valid-plan, rejection, clarification, and repair tests.

## Run

From the Project 2 directory, using the existing read-only Python installation:

```sh
../774_Dim_Red_Project/.venv/bin/python -m genmodel_agent.planning outputs/pathmnist_inspection/profile.json --out outputs/my_pathmnist_plan
../774_Dim_Red_Project/.venv/bin/python -m genmodel_agent.planning outputs/pbmc3k_inspection/profile.json --out outputs/my_pbmc_plan
```

A separately installed Project 2 environment can use `.venv/bin/python` instead.
Use `--objective` to override the saved objective, and `--total-minutes`,
`--max-epochs`, `--max-samples` to set application-owned ceilings.
Output directories must not already exist. Fresh planning requires the installed
Codex CLI signed in with ChatGPT, network access and available Codex allowance.
There is no API-key fallback. No Project 1 files or environments are modified.

## Data flow

1. Read the Project 2 profile; save an effective input snapshot and source hash.
2. Send compact statistics, the objective, compute evidence, catalog and limits
   to Codex. No image arrays, expression matrices, source paths or per-feature
   values are sent. Profile warnings and objective text may contain user text.
3. Require a structured response: ready, needs_clarification, or not_recommended.
4. Independently validate a ready plan. One validation-feedback repair is allowed
   (two calls maximum). Operational errors stop; there is no silent fallback plan.
5. Save `modeling_plan.json`, `modeling_plan.md`, each prompt, raw response,
   response schema, backend event log and validation history.

The model gets no shell, browser, editing, or delegation tools. The adapter uses
read-only sandboxing and ignores user configuration/project instruction loading
for the planner invocation. Its prompts contain supplied evidence only.

## What is adaptive

The LLM interprets the objective, selects one or two catalog candidates, explains
exclusions, chooses bounded hyperparameters, and proposes measurable success
criteria and provisional downstream use. A fixed catalog constrains what future
execution tools can support; it does not prescribe the same model every time.

The initial task vocabulary covers data generation, class balancing/augmentation,
and representation learning. Other tasks explicitly require clarification rather
than being advertised as implemented. The catalog has image VAE, image GAN, count VAE, and distinct planar_flow, radial_flow, and realnvp_flow options for continuous transformed expression. All six trainers are implemented. The old generic continuous_flow name remains an unavailable legacy placeholder. Validation rejects GAN/flow representation-learning plans. Diffusion is deferred;
the assignment does not require every family on every dataset.

## Validation boundaries

Enforced: exact plan fields; known model names; modality and likelihood compatibility;
raw-count provenance for cell workflows; complete categorical labels for conditional
models; official image splits; train-only fitted preprocessing; positive held-out
fractions for unsplit data; supported ranges; total model training-time ceilings;
objective-specific metrics and baselines; explicit model exclusions.

Not established by validation: scientific correctness of prose; whether the LLM
perfectly interpreted the user's intent; actual RAM/runtime feasibility; availability
of unseen official validation/test files; quality of generated data. Numerical
thresholds proposed by the LLM are hypotheses, not class requirements or empirically
calibrated guarantees. These need execution-stage checks and substantive review.
The time ceiling currently bounds planned training allocations, not measured total
pipeline duration. Compact profiles omit per-class counts but retain label cardinality
and imbalance summaries. The local validator uses full available label counts.

`ready` means the proposal passed current planning checks. New plans set `execution_ready` to true only for implemented trainer selections, subject to source and resource checks at execution. Archived plans predate the trainers and retain their historical false flag; the executor revalidates rather than trusting this flag. Downstream uses remain provisional because quality evaluation is pending.

## Execution stage

The image/count VAE tools, split handling, train-only preprocessing, checkpoints, resource monitoring and generated outputs are now implemented. See [execution details](execution_implementation.md). Metric contracts and evaluation-driven recommendations are now implemented; see [the evaluation guide](evaluation_implementation.md).

## Live verification on 2026-10-05

All 14 tests passed. Three actual Codex planning calls passed validation on their
first response (no repair). The initial sandbox-restricted call failed to initialize
Codex and was preserved in `outputs/pathmnist_plan`; it was not a modeling failure.

| Experiment | Selected model | Objective-dependent behavior |
|---|---|---|
| PathMNIST augmentation | Conditional image VAE | 4,500 labeled samples; real-only classifier baseline; macro-F1 and class recall checks |
| PBMC generation | Negative-binomial count VAE | 1,000 training-selected genes; 2,000 synthetic profiles; distribution/diversity/overlap checks |
| PathMNIST representation | Unconditional image VAE | PCA baseline; representation evaluation; 512 diagnostic samples |

Saved outputs: `outputs/pathmnist_plan_live`, `outputs/pbmc3k_plan_live`, and
`outputs/pathmnist_representation_plan_live`. Model family need not change when
objectives change: conditioning, generation purpose, budgets and evaluation did.
These three examples demonstrate responsiveness, not comprehensive robustness.

See [normalizing-flow implementation](normalizing_flows_implementation.md) for flow-specific planner fields, constraints, and verification.
