"""Versioned, executable metric meanings and conservative interpretation gates."""
VERSION = "2.0"
CONTRACT = {
    "version": VERSION,
    "validity": "Finite samples, exact expected shape, image uint8 range or nonnegative integer counts, label/gene alignment.",
    "distribution_fidelity": {
        "image": "Mean marginal Wasserstein distance in training-fitted standardized 4x4 pooled pixel/channel-variation features; includes per-class checks when labels exist. Not FID or a biological/perceptual score.",
        "counts": "Mean absolute differences in log1p gene means, log1p gene variances, zero fractions, log1p panel-total Wasserstein distance, and correlations among up to 64 training-variable genes.",
        "reference": "Equal-sized training resampling and held-out real observations, ten fixed-seed reference draws; report median and reference range, not a confidence interval.",
        "screen": "Synthetic discrepancy <= 1.1 times median real-reference discrepancy, with absolute tolerance 1e-8. A heuristic screen, not a calibrated scientific acceptance guarantee."},
    "diversity": "Median pairwise Euclidean distance in training-standardized pooled-image/log1p-count features; compare to equal-sized held-out samples; ratio screen 0.9..1.1.",
    "nearest_neighbor_overlap": "Nearest distances to a fixed training subset; near threshold is its leave-one-out 1st percentile. Report synthetic and held-out rates, plus exact duplicate rates against the same training subset. No privacy guarantee.",
    "real_test_macro_f1": "Real-only vs real+synthetic fixed C=1 logistic regression on training-fitted pooled-image features. Validation mode computes a validation proxy; only final mode computes test scores. Include class-matched real-resampling augmentation control.",
    "per_class_recall": "Recall on the same held-out observations with fixed training-class vocabulary; paired, class-stratified bootstrap for macro-F1 differences.",
    "heldout_representation_quality": "Frozen unconditional VAE posterior means vs training-fitted PCA at the same latent dimension; training-standardized embeddings with identical logistic probes and held-out labels. Unlabeled data are explicitly unsupported for this utility test.",
    "decision_rules": "Evidence policy v2: invalid output do_not_use; pilots, validation and incomplete evidence insufficient_evidence. No conjunction of quality thresholds establishes usability. Frozen converged task comparison with positive paired interval (and positive comparison against resampling for augmentation) supports task-scoped limited_use; a negative interval supports task-scoped do_not_use. Otherwise insufficient_evidence. No absolute F1 improvement threshold. Generation without a measured downstream task cannot establish biological usefulness.",
    "image_assessment": "Supplementary joint empirical energy distances in training-standardized local-gradient mean/std and training-only pixel PCA (up to 32 components); reference draws and conditional bootstrap intervals, without cutoffs. Fixed-seed per-class real/generated visual packets record source indices and explicit review status. Not FID or a validated histology score.",
    "utility_gate": "Legacy descriptive flag only, not the recommendation gate: Macro-F1 improvement >=0.01, paired 95% bootstrap lower bound >0, and no class recall loss >0.02; all arms converged. Representation compares VAE to PCA.",
}
DEFAULTS = {"seed": 42, "max_train": 4096, "max_holdout": 1024, "max_generated": 2048,
            "metric_rows": 512, "bootstrap": 200, "reference_repeats": 10}
