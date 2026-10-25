# Evaluation and evidence-based recommendations

Historical v1 implementation reference. The current [v2 policy and image extensions](evidence_evaluation_v2.md) supersede the decision gates below; the original metric definitions remain available.

The evaluator reads a saved execution folder and writes `evaluation_results.json`,
`evaluation_report.md`, and per-model figures/diagnostic arrays. It uses validation
observations by default. All work and the Python environment are inside Project 2.

## Commands

```sh
.venv/bin/python -m genmodel_agent.evaluation outputs/pathmnist_training_mps_pilot --out outputs/new_image_evaluation
.venv/bin/python -m genmodel_agent.evaluation outputs/pbmc3k_training_mps_pilot --out outputs/new_cell_evaluation
.venv/bin/python -m genmodel_agent.evaluation outputs/pathmnist_representation_training_pilot --out outputs/new_representation_evaluation
```

Default behavior includes online Codex interpretation after numerical evaluation.
Use `--no-ai` for fully local metrics and deterministic recommendations. AI failures
are recorded without discarding the numerical report. A successful numerical run
can therefore have an unavailable AI narrative; inspect its status separately.
Use `--dataset` only to relocate the identical input data, verified by source hashes.
Choose a new output directory for every evaluation.

## Code map

| File | Purpose |
|---|---|
| `evaluation_contract.py` | Versioned metric definitions, sample caps and decision rules |
| `evaluation_data.py` | Source/checkpoint verification, recorded split loading and training-only evaluator fitting |
| `evaluation_metrics.py` | Validity, fidelity, diversity, duplicates/neighbors and paired predictive scores |
| `evaluation.py` | Orchestration, final-test gate, reports, plots and bounded AI interpretation |
| `tests/test_evaluation.py` | Reference cases, test isolation, freeze/mutation checks, missing artifacts and interpretation constraints |

## Exact metric definitions

Image features are a fixed 4×4 spatial pooling grid plus channel standard deviations,
standardized using the recorded real training subset only. Fidelity is mean marginal
1D Wasserstein distance in these features. This is a lightweight image diagnostic,
not FID and not a validated histological perceptual score. Conditional outputs also
receive per-class diagnostics. Classes with fewer than three comparison observations
are reported as unassessable rather than assigned a passing score.

PBMC fidelity uses the fitted gene panel and compares log1p gene means, log1p gene
variances, zero fractions, log1p panel-total distributions, and correlations among
up to 64 genes ranked by training log-count variance. Correlations with zero-variance
columns use a zero-vector convention; these are descriptive diagnostics. Raw counts
are used for means/variances/zeros; log1p counts are used for correlation and distance
features. No new gene selection is fitted on validation or test cells.

Discrepancies are compared to ten fixed-seed, equally sized training-resampling
references evaluated against the same held-out observations. The report shows the
reference median and range (not a confidence interval), plus bootstrap intervals
for the synthetic discrepancy. Distribution screening uses <=1.1 times the median
reference discrepancy with a 1e-8 absolute tolerance. This is a transparent heuristic,
not an empirically calibrated scientific acceptance rule.

Diversity is median pairwise Euclidean distance in training-standardized features,
compared to equally sized held-out samples. The illustrative ratio screen is 0.9–1.1.
Overlap uses nearest distances to the recorded training reference subset, with a
threshold at the first percentile of leave-one-out training distances. Exact row
matches are also counted. Counts are canonicalized numerically for duplicate checks.
The overlap screen requires synthetic exact-duplicate rate no larger than the held-out
rate and near-match rate no more than two percentage points higher. Training resampling
has exact-match rate 1 by construction. None of these checks guarantees privacy.

## Objective-specific comparisons

For image augmentation, identical C=1 logistic classifiers fit real-only versus
real-plus-synthetic pooled-image features. All scaling fits real training data only.
A class-matched real-resampling augmentation control is included to distinguish
adding examples from adding generative information. Both arms have the same solver,
tolerance and iteration cap; convergence is recorded. This is a lightweight linear
baseline, not a state-of-the-art image classifier.

For representation learning, unconditional VAE posterior means are compared with
training-fitted PCA at the planned latent dimension (bounded by training dimensions).
Both use training-fitted embedding standardization and the same logistic probe.
Held-out labels are used only for scoring, never encoder inputs. Unlabeled representation
utility is explicitly unavailable. No invented cell-type labels or cluster truth are used.

Macro-F1 and per-class recall use the fixed training vocabulary. Differences receive
paired, class-stratified 95% bootstrap intervals on the same held-out observations.
The utility screen requires an absolute macro-F1 improvement of at least .01, a lower
interval bound above zero, no recall decrease exceeding .02, and solver convergence.
Bootstrap intervals condition on one trained generator/classifier pair; they do not
capture variation across training seeds. Validation scores are explicitly named proxies,
not test performance.

## Recommendations and AI

Rules assign `do_not_use`, `insufficient_evidence`, or `limited_use`. Invalid samples
are rejected. Pilots and validation-only runs remain insufficient evidence for adoption.
Final nonpilot evaluation also needs sufficient samples, complete applicable checks,
and a compatible executable plan contract. Failing final screens prevent a recommendation
for use. Passing screens justify only restricted use in the evaluated setting.

Codex receives numerical results, limits and the rule decision. Its structured response
must cite existing evidence sections, cover every model and preserve that decision.
An attempted upgrade is rejected. AI prose is explanatory and not mechanically verified
claim-by-claim; the JSON numerical evidence remains authoritative. No image/count arrays
are sent to Codex, and the interpretation call has no execution/editing tools.

New AI plans receive the exact versioned evaluator contract. Historical saved plans
contain free-form numerical criteria that cannot safely be inferred as executable code.
Their reports explicitly state that the new diagnostics do not fully verify those older
narrative rules. This prevents a false claim that all old acceptance conditions passed.

## Final test protection

After model and evaluation choices are fixed, a nonpilot validation run can save a protocol:

```sh
.venv/bin/python -m genmodel_agent.evaluation outputs/full_run --out outputs/full_validation --freeze outputs/frozen_protocol.json
.venv/bin/python -m genmodel_agent.evaluation outputs/full_run --out outputs/final_test --mode test --protocol outputs/frozen_protocol.json
```

The protocol records execution/checkpoint/sample/split hashes, evaluator code, package
versions, options and signatures of training-fitted transforms/classifiers. Final mode
refits only on the same training data and verifies those signatures before loading test
values. An atomic `final_test_claim.json` in the execution folder prevents accidental
repeat test consumption. The claim stays after a later failure. Pilot runs cannot access
final test data. This is a workflow safeguard, not an adversarial security boundary:
manually deleting records or creating new copies can bypass local bookkeeping.

Defaults cap training references at 4,096, held-out observations at 1,024, generated
observations at 2,048, and pairwise metric samples at 512. Exact selected indices are
saved. These caps bound routine work on this Mac; results apply to those subsets.
Full original sample validity is checked before selecting an evaluation subset.

## Pilot findings

All three existing pilot evaluations completed on validation observations. Each had
512 training observations, 128 validation observations and 32 generated samples.

| Pilot | Measured validation finding |
|---|---|
| PathMNIST augmentation | Real-only macro-F1 0.5950; augmented 0.5441; delta −0.0509; paired 95% interval approximately [−0.1062, 0.0033] |
| PathMNIST representation | PCA macro-F1 0.4072; VAE 0.3340; delta −0.0731; paired 95% interval approximately [−0.1497, 0.0044] |
| PBMC generation | Mean, variance, zero-fraction and correlation discrepancies exceeded real-reference screens; diversity ratio approximately 6.07 |

Both predictive intervals include zero. These small, two-epoch pilots do not establish
that either approach helps or harms general downstream performance. All recommendations
remain `insufficient_evidence`. No actual PathMNIST/PBMC test values were evaluated.

References: [SciPy Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html),
[scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
[logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

## Verification

The evaluation tests cover invalid numeric outputs and gene IDs, distribution shifts,
exact copying, paired scores, validation/test isolation, frozen-protocol mismatch,
repeat test access, pilot test refusal, missing artifacts, and AI decision/evidence
validation. The integration final-test checks use synthetic test fixtures only.
All three real pilot reports now include successful AI interpretations. One response
used valid dotted evidence references that the original validator rejected; it was
revalidated from the saved response after adding checked nested-path support.
No additional AI call was needed, and the initial rejection record was preserved.

Pilot paths retain the exact code fingerprints from when numerical metrics were
computed. Later validator hardening does not change those measured values. A future
final evaluation must use a newly frozen protocol matching the then-current code.
