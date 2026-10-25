# Completed evidence-based evaluation changes

All five agreed changes are implemented: separate validity/fit/quality/utility;
model-specific training summaries; an evidence-based recommendation policy;
stronger PathMNIST assessments; and output-domain-specific PBMC interpretation.

## Recommendation policy version 2

Invalid generated output is rejected. Pilots and validation runs remain development
evidence. Their narrative now states whether a measured task interval favors the
alternative, favors the baseline, or remains inconclusive. A conjunction of heuristic
quality screens no longer determines adoption. Statistical generation without a
measured downstream task cannot establish biological usefulness, even when every
statistical flag is favorable.

For a compatible frozen final evaluation with a converged task comparison, a paired
95% macro-F1 interval above zero supports only task-scoped limited use. Augmentation
also needs a positive comparison against matched real resampling. A wholly negative
interval supports not using the alternative for that particular task. Overlapping-zero
or missing intervals yield insufficient evidence. Per-class tradeoffs are reported.
The former 0.01 F1 and 0.02 recall cutoffs remain descriptive legacy flags, not adoption
gates. Distributional concerns accompany any task-specific recommendation rather
than being erased by improved classifier performance.

This is a documented operational policy, not a theorem from the lecture PDFs.
Intervals condition on one trained generator/classifier; they do not measure seed
variability or establish clinical effectiveness. The bounded revision loop can stop
on supported validation utility even when quality flags persist. For generation-only
objectives, heuristic screens still guide whether to request a bounded revision;
that stopping signal does not grant permission for downstream use.

## Image evidence

The original pooled-feature diagnostics and augmentation classifiers are retained
for comparison. New supplementary feature spaces capture information they omit:

- Gradient magnitude mean and standard deviation within each cell of a 4x4 grid,
  separately for each channel. These expose local contrast/texture differences.
- Up to 32 PCA components fitted to complete real training images. This captures
  spatial covariance beyond independent pooled-feature distributions.

Scalers and PCA fit only the recorded real training subset. Their signatures are
included in the frozen protocol. No pretrained weights or new dependencies are
needed. These handcrafted gradients are not canonical HOG; neither feature space
is FID, a histology-trained representation, or evidence of biological validity.

Each feature space uses the joint empirical energy statistic
`2 mean(||X-Y||) - mean(||X-X'||) - mean(||Y-Y'||)` (including empirical self pairs).
This is the squared-distance convention corresponding to the square-root energy
distance documented by SciPy in one dimension. Unlike averaging marginal
Wasserstein distances, it can respond to dependence differences within the chosen
feature space. It still does not capture all image information.

Reports show the discrepancy, conditional bootstrap interval, and matched-size
training-resampling reference median/range. Reference ranges are not confidence
intervals. No new image acceptance cutoff is introduced. Exact selected local
indices are recorded; full subset-to-source mappings remain in evaluation_indices.

Fixed-seed sample grids show up to four real and four generated images per class,
with original indices, nearest-neighbor pixel display and no cherry-picking.
Unconditional outputs are not assigned guessed labels. A generated packet is marked
not-yet-reviewed; only a separate documented visual inspection counts as review.
The current fuller VAE/GAN artifacts were inspected by the assistant, not a
histopathologist. The observations cannot establish tissue-label correctness.

## PBMC scope

Count-VAE checks continue to use integer counts, count moments, zeros, totals and
correlations. Flow checks use continuous log-dequantized selected-gene expression;
integer validity and zero-count claims do not apply. Neither passing moments nor
matching correlations establishes biological usefulness. Flow NLL remains in its
standardized continuous coordinates, with comparisons restricted to matching
features and preprocessing.

## Reproducibility and review

`evaluation_contract.py` is now version 2; `evidence_policy.py` implements the
conclusion rules and `image_assessment.py` implements the added image evidence.
History files now join the artifact manifest. Prior reports and checkpoints remain
unchanged. Contract mismatch is explicit for older plans: a new evaluation does
not silently approve their earlier narrative criteria. A new frozen protocol is
required before any final test. No real final-test data were accessed here.

Review the new validation reports:

- [GAN](../outputs/pathmnist_gan_evidence_v3/evaluation_report.md)
- [VAE](../outputs/pathmnist_vae_evidence_v3/evaluation_report.md)

The initial v2 output folders are retained as intermediate verification records;
v3 uses improved grid spacing and the final report format. No generators were retrained.

References supporting the components, not the particular policy:

- [PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Energy distance definition](https://docs.scipy.org/doc/scipy-1.14.0/reference/generated/scipy.stats.energy_distance.html)
- [Gradient-based image descriptors](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html) explains the motivation; our local magnitude summaries are explicitly simpler than HOG.
- Supplied VAE Sections 16.3–16.5, GAN Section 18, and flow Section 19.2 support the method-fit diagnostics, not the custom decision policy.

Additional updated validation reports: [PBMC count VAE](../outputs/pbmc_count_evidence_v2/evaluation_report.md), [PBMC flows](../outputs/pbmc_flows_evidence_v2/evaluation_report.md), and [revised flow](../outputs/pbmc_flow_revision_evidence_v2/evaluation_report.md). The full 70-test suite passed, followed by an additional targeted pipeline-policy regression test. All 79 Project 1 snapshot files remain unchanged.
