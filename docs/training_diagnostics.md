# Training diagnostics and evidence-aware revisions

Update: the later [evaluation v2 milestone](evidence_evaluation_v2.md) completes
the recommendation-policy and image-evaluation changes. References below to
unchanged final-test decision gates describe the earlier diagnostics milestone.

The agent now reports four separate questions: output validity, model fit,
sample quality, and downstream utility. Training diagnostics inform a revision;
they never override measured quality or establish usefulness by themselves.

`genmodel_agent/diagnostics.py` summarizes recorded histories without reading
raw observations or test data. It reports first/last/minimum values and recent
changes over up to five recorded epochs. VAE summaries include negative ELBO,
reconstruction NLL and KL; flows include continuous-coordinate NLL; GANs include
adversarial losses and the recorded validation feature discrepancy.

With at least five epochs, falling training loss and rising validation loss
produce a possible-overfitting hypothesis. A decreasing recent validation loss
whose minimum occurs at the final epoch produces a still-improving hypothesis.
These are descriptive signs, not statistical tests, causal findings or universal
thresholds. Missing or nonfinite histories do not produce confident diagnoses.
No KL threshold automatically diagnoses posterior collapse, and GAN loss scales
do not automatically diagnose imbalance or mode collapse.

The planner receives both these summaries and unchanged numerical sample-quality
and utility evidence. Its guidance asks for metric/epoch evidence, a predicted
effect and competing explanation, and preferably one or two scientific setting
changes at a time. These reasoning requirements guide the AI; their scientific
truth is not mechanically guaranteed. Existing schema validation, budget limits,
split preservation and prevention of identical retries remain enforced.

Reports label the existing quality thresholds as exploratory workflow controls.
They separate fit from sample quality and downstream evidence. Existing screening
numbers and guarded final-test policy remain unchanged: no thresholds were relaxed
to obtain a passing result. No unsupported scientific replacement cutoff has been
introduced. A screen failure means concern under this policy, not universal
unusability; no downstream experiment means benefit remains unmeasured.

The lecture foundations are VAE Sections 16.3–16.5 (ELBO, reconstruction and KL,
and limits of ELBO as a distribution-matching objective), GAN Section 18
(adversarial objectives), and normalizing-flow Section 19.2 (likelihood).
The trend heuristics and four-part reporting structure are implementation choices,
not rules prescribed in the lectures. Flow likelihood comparisons require matching
features and preprocessing and do not represent raw-count likelihoods.

Both one-command workflow feedback and standalone evaluations include these
summaries. Earlier saved reports are unchanged. The additive retrospective preview
at `outputs/diagnostics_review_v1/review.md` uses the completed PBMC flow experiment
without retraining, recalculating metrics, accessing test observations, or making
new AI calls. New full evaluations need a newly frozen protocol because the code
signature now includes the diagnostic implementation.

This change does not add image feature extractors, biological utility tests, or
new generator architectures. It makes existing evidence explicit and usable in
future bounded revision decisions.
