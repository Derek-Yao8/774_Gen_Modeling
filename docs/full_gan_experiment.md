# Fuller PathMNIST GAN validation experiment

This experiment was requested to assess the implemented GAN beyond its two-epoch
pilot. The GAN family was specified by the user; the AI selected its numerical
settings. This is not evidence that an unrestricted planner preferred GAN over
VAE. WGAN/WGAN-GP, flows and diffusion remain deferred.

## Recorded design

The complete workflow was launched with one command, using the existing trainer
and evaluation contracts:

```sh
.venv/bin/genmodel-run ../774_Dim_Red_Project/data/pathmnist.npz --split train --objective "Run a fuller conditional image GAN experiment on PathMNIST: generate labeled images and evaluate synthetic augmentation on held-out real validation images. Use the image GAN requested by the user as the alternative to the existing VAE experiment, without pilot subset caps. Choose suitable training hyperparameters within the supplied budget; preserve official splits and seed 42 for comparison. Generate 5000 images if feasible within the resource contract. Do not infer usefulness from training losses or assume the GAN is better than the VAE." --out outputs/pathmnist_gan_full_workflow_v1 --device mps --total-minutes 30 --max-rounds 1 --max-epochs 40 --max-samples 5000 --max-memory-gib 6
```

The selected configuration is conditional `image_gan`, latent dimension 64,
hidden dimension 128, batch size 64, learning rate 0.0002, up to 40 epochs,
patience eight, and 5,000 generated images. The GAN uses its existing separate
Adam optimizers with betas (0.5,0.999) and non-saturating BCE objectives. It is
not a Wasserstein GAN; Wasserstein distance is used for checkpoint selection
and evaluation only.

Training uses all 89,996 official training images, with official validation
reserved for checkpoint selection and development evaluation. The checkpoint
selector uses a fixed subset of at most 512 validation images, not every
validation image each epoch. The trainer reserves 20% of its 30-minute allowance
for finishing outputs. Early stopping and resource limits can end training before
40 epochs. The monitored 6 GiB memory limit is not an OS reservation.

Only one training attempt is authorized for this experiment's workflow run.
A failed quality screen ends the run with recorded findings rather than initiating
an unbounded search. A successful computation does not imply useful synthetic data.

## Comparison scope

The comparison target is the saved non-pilot VAE experiment in
`outputs/pathmnist_training_validation_v1` and its corresponding evaluation.
Both model families use the same final image-evaluation contract. The evaluator
caps real training references at 4,096 images, real validation at 1,024, generated
samples at 2,048, and equal-size quality comparisons at 512. Evaluation records
will be checked for matching settings and real-data indices before reporting
side-by-side results.

This is a descriptive comparison of two single trained configurations, not a
multi-seed benchmark or a causal isolation of model family. Architectures,
optimizer settings and checkpoint criteria differ. Training losses are not
comparable between VAE and GAN. Paired bootstrap intervals in each evaluation
condition on its trained classifier/model and do not measure training variability.
No new GAN-versus-VAE significance claim follows merely from comparing their
individual intervals.

The final test set remains reserved. Source data in Project 1 are read-only;
all new artifacts are saved under Project 2. No acceptance thresholds or model
architecture were changed for this experiment.

## Results

Training completed successfully on MPS after 21 full epochs. Early stopping
triggered after eight epochs without improvement over epoch 13, whose checkpoint
score was 0.254423. The selected checkpoint generated all 5,000 requested images.
Supervised wall time was 1,060.4 seconds (17.7 minutes), with peak process RSS
0.74 GiB; this is not total unified/GPU memory. No resource limit was triggered.

The one-command workflow also completed evaluation and AI interpretation. Its
terminal status is `round_limit_reached`: the configured one experiment finished
with failed validation screens. This is a normal scientific stopping outcome,
not a software/training failure.

| Validation measure | Saved VAE | Fuller GAN |
|---|---:|---:|
| Fidelity discrepancy / real-reference median (screen <= 1.1) | 4.441 | 3.017 |
| Diversity ratio (screen 0.9–1.1) | 0.725 | 0.772 |
| Real-only classifier macro-F1 | 0.6337 | 0.6337 |
| Real + synthetic classifier macro-F1 | 0.5804 | 0.6215 |
| Synthetic augmentation minus real-only | -0.0532 | -0.0122 |
| Paired 95% bootstrap interval for that difference | [-0.0697, -0.0313] | [-0.0384, 0.0143] |
| Real + class-matched resampling macro-F1 | 0.6514 | 0.6514 |
| Overall quality / augmentation utility screens | Fail / Fail | Fail / Fail |

The GAN's measured fidelity, diversity and augmentation macro-F1 were closer to
the desired outcomes than the saved VAE's. However, all nine GAN per-class quality
screens failed, and there is no demonstrated augmentation benefit over real-only
training: the GAN's paired difference interval includes zero. Its augmentation
also performed worse than the real-resampling control in this fitted comparison:
delta -0.02987, conditional interval [-0.05233, -0.00752]. The control's higher
aggregate macro-F1 alone does not establish it passes every per-class utility
requirement.

All 5,000 generated observations passed shape/dtype/label validity checks. Counts
were balanced to rounding: five classes had 556 images and four had 555. The
separate evaluation stages use explicit recorded subsets: validity checks all
5,000, evaluation retains 2,048 generated images, distribution comparisons use
512 matched rows, and classifier utility is measured on 1,024 real validation
images with 4,096 real training references. The AI interpretation notes that its
compact input did not explain this relationship; these are deliberate evaluator
caps, not contradictory sample counts.

No exact duplicates or near matches were detected against the recorded training
reference subset. This is not a privacy guarantee. The generated grid shows
more varied structures than the early pilot, but visual appearance does not
establish label fidelity, histological validity or analytical usefulness.

## Recommendation

Do not adopt this synthetic augmentation for downstream training on the evidence
available. The reported label remains `insufficient_evidence` because final
testing is reserved; the failed quality/utility checks are also explicit. Keep
this as a completed negative/mixed validation experiment. Additional tuning is
optional and should require a specific rationale and new budget; it is not
necessary to keep training until a model passes. No new model families or
architecture revisions were added during this run.

## Verification and review

Training and validation indices match the earlier VAE experiment exactly.
Evaluation train, held-out and generated-array indices also match, as do the
evaluation contract, options, software versions, and hashes of the metric,
evaluation-data and contract modules. Matching generated indices identify the
same sampling positions, not identical synthetic contents.

Checkpoint/output artifacts and class counts were checked; all computations
completed and the test-access flag is false. All 79 recorded Project 1 source and
document hashes match the pre-run snapshot. This experiment changed no model or
evaluation source code, so no additional unit-test run was needed; the GAN's
previous implementation milestone passed 51 tests.

- [Combined agent report](../outputs/pathmnist_gan_full_workflow_v1/generated_report.md)
- [AI-selected plan](../outputs/pathmnist_gan_full_workflow_v1/round_1/planning/modeling_plan.md)
- [Training record](../outputs/pathmnist_gan_full_workflow_v1/round_1/execution/execution_report.md)
- [Generated images](../outputs/pathmnist_gan_full_workflow_v1/round_1/execution/image_gan/generated_grid.png)
- [Evaluation and AI interpretation](../outputs/pathmnist_gan_full_workflow_v1/round_1/evaluation/evaluation_report.md)
- [Machine-readable comparison and verification](../outputs/pathmnist_gan_full_workflow_v1/experiment_verification_and_comparison.json)
