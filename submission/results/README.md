# Saved final validation results

Start with [generated_report_1](generated_report_1.md) (PathMNIST) and
[generated_report_2](generated_report_2.md) (PBMC). Figures are embedded through
relative links; keep this directory intact. Markdown renders directly on GitHub
or in a Markdown preview. These are validation results, not final-test assessments.

Authoritative PathMNIST round 1 is `pathmnist_recovery_v1/round_1`; round 2 is
`pathmnist_continuation_v1`. PBMC uses `pbmc/round_1` and `pbmc/round_2`.
The PathMNIST report explains the interruption and planner usage limit. Earlier
failure statuses retained in historical records do not describe the completed set.

Each model's `execution/<model>/generated.npz` contains synthetic outputs.
Plans, effective hyperparameters, histories, split indices, evaluation arrays,
metrics, recommendations and AI interpretations are included. Checkpoints and
verbose CLI event streams are omitted; they are not required to inspect submitted
results. Historical execution reports describe the original checkpoint files;
those files remain in the local experiment archive. Regenerate them by training
if a fresh checkpoint-based evaluation is needed.

`experiment_config.json`, execution jobs and provenance preserve original paths
as historical records. Do not run them as new configurations. For new runs use
`../run_config.json` and `../../scripts/run_submission.py`, which use Project 2's
local data paths and fresh output directories. `source_snapshot.zip` preserves
the modeling code used for the experiments. The current root source additionally
contains the later general-report visualization update.

`verification.json` is the historical experiment check; `../manifest.json` lists
checksums of the portable package after Markdown image links were rewritten.
The separate visual-review appendix records post-run assistant observations,
not expert review or a completed review by the automated interpreter.
